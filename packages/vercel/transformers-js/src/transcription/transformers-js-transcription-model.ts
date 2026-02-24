import {
  TranscriptionModelV3,
  TranscriptionModelV3CallOptions,
  SharedV3Warning,
  LoadSettingError,
} from "@ai-sdk/provider";
import {
  AutoTokenizer,
  AutoProcessor,
  WhisperForConditionalGeneration,
  full,
  type PreTrainedTokenizer,
  type Processor,
  type PreTrainedModel,
  type PretrainedModelOptions,
  type ProgressInfo,
  type Tensor,
} from "@huggingface/transformers";
import type { DownloadProgressCallback } from "@browser-ai/shared";

export type TransformersJSTranscriptionModelId = string;

export interface TransformersJSTranscriptionSettings extends Pick<
  PretrainedModelOptions,
  "device" | "dtype"
> {
  /**
   * Progress callback for model initialization
   */
  initProgressCallback?: DownloadProgressCallback;
  /**
   * Raw progress callback from Transformers.js
   */
  rawInitProgressCallback?: (progress: ProgressInfo) => void;
  /**
   * Maximum number of new tokens to generate
   * @default 64
   */
  maxNewTokens?: number;
  /**
   * Language hint for better transcription accuracy
   */
  language?: string;
  /**
   * Return timestamps for segments
   * @default false
   */
  returnTimestamps?: boolean;
  /**
   * Optional Web Worker to run the model off the main thread
   */
  worker?: Worker;
}

/**
 * Check if we're running in a browser environment
 */
export function isBrowserEnvironment(): boolean {
  return typeof window !== "undefined";
}

/**
 * Check if we're running in a server environment (Node.js)
 */
export function isServerEnvironment(): boolean {
  return typeof window === "undefined" && typeof process !== "undefined";
}

interface TranscriptionModelConfig extends TransformersJSTranscriptionSettings {
  modelId: TransformersJSTranscriptionModelId;
}

type TranscriptionModelInstance = [
  PreTrainedTokenizer,
  Processor,
  PreTrainedModel,
];

export class TransformersJSTranscriptionModel implements TranscriptionModelV3 {
  readonly specificationVersion = "v3";
  readonly provider = "transformers-js";
  readonly modelId: TransformersJSTranscriptionModelId;

  private readonly config: TranscriptionModelConfig;
  private modelInstance?: TranscriptionModelInstance;
  private isInitialized = false;
  private initializationPromise?: Promise<void>;
  private workerReady = false;

  constructor(
    modelId: TransformersJSTranscriptionModelId,
    options: TransformersJSTranscriptionSettings = {},
  ) {
    this.modelId = modelId;
    this.config = {
      modelId,
      device: "auto",
      dtype: "auto",
      maxNewTokens: 64,
      returnTimestamps: false,
      ...options,
    };
  }

  private async getSession(
    onInitProgress?: DownloadProgressCallback,
  ): Promise<TranscriptionModelInstance> {
    if (this.modelInstance && this.isInitialized) {
      return this.modelInstance;
    }

    if (this.initializationPromise) {
      await this.initializationPromise;
      if (this.modelInstance) {
        return this.modelInstance;
      }
    }

    this.initializationPromise = this._initializeModel(onInitProgress);
    await this.initializationPromise;

    if (!this.modelInstance) {
      throw new LoadSettingError({
        message: "Transcription model initialization failed",
      });
    }

    return this.modelInstance;
  }

  private async _initializeModel(
    onInitProgress?: DownloadProgressCallback,
  ): Promise<void> {
    try {
      const { device, dtype } = this.config;
      const progress_callback = this.createProgressTracker(onInitProgress);

      const resolvedDevice = this.resolveDevice(device);
      const resolvedDtype = this.resolveDtype(dtype);

      // Create model instance
      const [tokenizer, processor, model] = await Promise.all([
        AutoTokenizer.from_pretrained(this.modelId, { progress_callback }),
        AutoProcessor.from_pretrained(this.modelId, { progress_callback }),
        WhisperForConditionalGeneration.from_pretrained(this.modelId, {
          dtype: resolvedDtype,
          device: resolvedDevice,
          progress_callback,
        }),
      ]);

      this.modelInstance = [tokenizer, processor, model];

      // Warm up the model (skip in server environment to reduce initialization time)
      if (isBrowserEnvironment()) {
        try {
          const numMelBins = (model.config as any)?.num_mel_bins ?? 80;
          await model.generate({
            inputs: full([1, numMelBins, 3000], 0.0),
          });
        } catch (error) {
          // Ignore warmup errors
          console.warn("Model warmup failed:", error);
        }
      }

      onInitProgress?.(1.0);
      this.isInitialized = true;
    } catch (error) {
      this.modelInstance = undefined;
      this.isInitialized = false;
      this.initializationPromise = undefined;

      throw new LoadSettingError({
        message: `Failed to initialize TransformersJS transcription model: ${
          error instanceof Error ? error.message : "Unknown error"
        }`,
      });
    }
  }

  private resolveDevice(
    device?: PretrainedModelOptions["device"],
  ): PretrainedModelOptions["device"] {
    if (device && device !== "auto") {
      return device as PretrainedModelOptions["device"];
    }

    if (isServerEnvironment()) {
      return "cpu";
    }

    return "auto";
  }

  private resolveDtype(
    dtype?: string | object,
  ): PretrainedModelOptions["dtype"] {
    if (dtype && dtype !== "auto") {
      return dtype as PretrainedModelOptions["dtype"];
    }

    return {
      encoder_model: "auto",
      decoder_model_merged: "auto",
    };
  }

  private createProgressTracker(onInitProgress?: DownloadProgressCallback) {
    const fileProgress = new Map<string, { loaded: number; total: number }>();

    return (p: ProgressInfo) => {
      this.config.rawInitProgressCallback?.(p);

      if (!onInitProgress) return;

      // Type guard to check if p has model file property
      const progressWithFile = p as ProgressInfo & {
        file?: string;
        loaded?: number;
        total?: number;
      };
      const file = progressWithFile.file;

      if (!file) return;

      if (p.status === "progress" && file) {
        fileProgress.set(file, {
          loaded: progressWithFile.loaded || 0,
          total: progressWithFile.total || 0,
        });
      } else if (p.status === "done" && file) {
        const prev = fileProgress.get(file);
        if (prev?.total) {
          fileProgress.set(file, { loaded: prev.total, total: prev.total });
        }
      }

      // Calculate overall progress
      let totalLoaded = 0;
      let totalBytes = 0;
      for (const { loaded, total } of fileProgress.values()) {
        if (total > 0) {
          totalLoaded += loaded;
          totalBytes += total;
        }
      }

      if (totalBytes > 0) {
        onInitProgress(Math.min(1, totalLoaded / totalBytes));
      }
    };
  }

  private convertAudioToFloat32Array(
    audio: string | Uint8Array | ArrayBuffer,
  ): Promise<Float32Array> {
    return new Promise((resolve, reject) => {
      try {
        let audioData: Uint8Array;

        if (typeof audio === "string") {
          // Base64 encoded audio data
          const binaryString = atob(audio);
          audioData = new Uint8Array(binaryString.length);
          for (let i = 0; i < binaryString.length; i++) {
            audioData[i] = binaryString.charCodeAt(i);
          }
        } else if (audio instanceof Uint8Array) {
          audioData = audio;
        } else if (audio instanceof ArrayBuffer) {
          audioData = new Uint8Array(audio);
        } else {
          throw new Error("Unsupported audio format");
        }

        // Create an AudioContext to decode the audio
        // TODO: Make this work for server environment too.
        if (
          typeof AudioContext !== "undefined" ||
          typeof (window as any)?.webkitAudioContext !== "undefined"
        ) {
          const AudioContextClass =
            AudioContext || (window as any).webkitAudioContext;
          const audioContext = new AudioContextClass({ sampleRate: 16000 });

          audioContext
            .decodeAudioData(
              audioData.buffer.slice(
                audioData.byteOffset,
                audioData.byteOffset + audioData.byteLength,
              ),
            )
            .then((audioBuffer) => {
              const channelData = audioBuffer.getChannelData(0);
              resolve(new Float32Array(channelData));
            })
            .catch(reject);
        } else {
          // This is a simplified approach. TODO: proper audio decoding!!
          const float32Data = new Float32Array(audioData.length);
          for (let i = 0; i < audioData.length; i++) {
            float32Data[i] = (audioData[i] - 128) / 128.0;
          }
          resolve(float32Data);
        }
      } catch (error) {
        reject(error);
      }
    });
  }

  private getArgs({ audio, providerOptions }: TranscriptionModelV3CallOptions) {
    const warnings: SharedV3Warning[] = [];

    const transformersJSOptions = providerOptions?.["transformers-js"];

    const language =
      typeof transformersJSOptions?.language === "string"
        ? transformersJSOptions.language
        : this.config.language;
    const returnTimestamps =
      typeof transformersJSOptions?.returnTimestamps === "boolean"
        ? transformersJSOptions.returnTimestamps
        : this.config.returnTimestamps;
    const maxNewTokens =
      typeof transformersJSOptions?.maxNewTokens === "number"
        ? transformersJSOptions.maxNewTokens
        : this.config.maxNewTokens;

    return {
      audio,
      language,
      returnTimestamps,
      maxNewTokens,
      warnings,
    };
  }

  /**
   * Check the availability of the TransformersJS transcription model
   */
  public async availability(): Promise<
    "unavailable" | "downloadable" | "available"
  > {
    // If using a worker (browser only), reflect worker readiness instead of main-thread state
    if (this.config.worker && isBrowserEnvironment()) {
      return this.workerReady ? "available" : "downloadable";
    }

    if (isServerEnvironment() && this.config.worker) {
      // Ignore worker config on server and use main thread
    }

    if (this.isInitialized) {
      return "available";
    }

    return "downloadable";
  }

  /**
   * Creates a session with download progress monitoring
   */
  public async createSessionWithProgress(
    onDownloadProgress?: DownloadProgressCallback,
  ): Promise<TransformersJSTranscriptionModel> {
    // If a worker is provided and we're in browser environment, initialize the worker
    // (and forward progress) instead of initializing the model on the main thread
    // to avoid double-initialization/downloads.
    if (this.config.worker && isBrowserEnvironment()) {
      await this.initializeWorker(onDownloadProgress);
      return this;
    }

    // In server environment or when no worker is provided, use main thread
    if (!this.initializationPromise) {
      this.initializationPromise = this._initializeModel(onDownloadProgress);
    }
    await this.initializationPromise;
    return this;
  }

  async doGenerate(
    options: TranscriptionModelV3CallOptions,
  ): Promise<Awaited<ReturnType<TranscriptionModelV3["doGenerate"]>>> {
    const currentDate = new Date();
    const { audio, language, returnTimestamps, maxNewTokens, warnings } =
      this.getArgs(options);

    // Use worker if provided and in browser environment
    if (this.config.worker && isBrowserEnvironment()) {
      return this.doGenerateWithWorker(
        audio,
        language,
        returnTimestamps,
        maxNewTokens,
        warnings,
        currentDate,
        options,
      );
    }

    // Main thread generation (browser without worker or server environment)
    const [tokenizer, processor, model] = await this.getSession(
      this.config.initProgressCallback,
    );

    try {
      const audioFloat32 = await this.convertAudioToFloat32Array(audio);

      const durationInSeconds = audioFloat32.length / 16000;

      const inputs = await processor(audioFloat32);

      const outputs = await model.generate({
        ...inputs,
        max_new_tokens: maxNewTokens,
        language: language,
        return_timestamps: returnTimestamps,
      });

      const transcriptionText = tokenizer.batch_decode(outputs as Tensor, {
        skip_special_tokens: true,
      })[0];

      return {
        text: transcriptionText,
        segments: [], // TODO: Parse segments from output if returnTimestamps is true
        language: typeof language === "string" ? language : undefined,
        durationInSeconds: durationInSeconds,
        warnings,
        response: {
          timestamp: currentDate,
          modelId: this.modelId,
          headers: {},
          body: JSON.stringify({ text: transcriptionText }),
        },
      };
    } catch (error) {
      throw new Error(
        `TransformersJS transcription failed: ${
          error instanceof Error ? error.message : "Unknown error"
        }`,
      );
    }
  }

  private async doGenerateWithWorker(
    audio: string | Uint8Array | ArrayBuffer,
    language: string | undefined,
    returnTimestamps: boolean | undefined,
    maxNewTokens: number | undefined,
    warnings: SharedV3Warning[],
    currentDate: Date,
    options: TranscriptionModelV3CallOptions,
  ): Promise<Awaited<ReturnType<TranscriptionModelV3["doGenerate"]>>> {
    const worker = this.config.worker!;

    await this.initializeWorker();

    const audioFloat32 = await this.convertAudioToFloat32Array(audio);
    const durationInSeconds = audioFloat32.length / 16000;

    const result = await new Promise<string>((resolve, reject) => {
      const onMessage = (e: MessageEvent) => {
        const msg = e.data;
        if (!msg) return;
        if (msg.status === "complete" && Array.isArray(msg.output)) {
          worker.removeEventListener("message", onMessage);
          resolve(String(msg.output[0] ?? ""));
        } else if (msg.status === "error") {
          worker.removeEventListener("message", onMessage);
          reject(new Error(String(msg.data || "Worker error")));
        }
      };
      worker.addEventListener("message", onMessage);
      worker.postMessage({
        type: "generate",
        data: {
          audio: Array.from(audioFloat32), // Convert to array for worker transfer
          language,
          maxNewTokens,
        },
      });

      if (options.abortSignal) {
        const onAbort = () => {
          worker.postMessage({ type: "interrupt" });
          options.abortSignal?.removeEventListener("abort", onAbort);
        };
        options.abortSignal.addEventListener("abort", onAbort);
      }
    });

    return {
      text: result,
      segments: [], // TODO: Parse segments from output if returnTimestamps is true
      language: typeof language === "string" ? language : undefined,
      durationInSeconds: durationInSeconds,
      warnings,
      response: {
        timestamp: currentDate,
        modelId: this.modelId,
        headers: {},
        body: JSON.stringify({ text: result }),
      },
    };
  }

  private async initializeWorker(
    onInitProgress?: DownloadProgressCallback,
  ): Promise<void> {
    if (!this.config.worker) return;

    // If already ready, optionally emit completion progress
    if (this.workerReady) {
      if (onInitProgress) onInitProgress(1);
      return;
    }

    const worker = this.config.worker;

    await new Promise<void>((resolve, reject) => {
      const trackProgress = this.createProgressTracker(onInitProgress);

      const onMessage = (e: MessageEvent) => {
        const msg = e.data;
        if (!msg) return;

        // Forward raw download progress events coming from @huggingface/transformers running in the worker
        if (msg && typeof msg === "object" && "status" in msg) {
          if (msg.status === "ready") {
            worker.removeEventListener("message", onMessage);
            this.workerReady = true;
            if (onInitProgress) onInitProgress(1);
            resolve();
            return;
          }
          if (msg.status === "error") {
            worker.removeEventListener("message", onMessage);
            reject(
              new Error(String(msg.data || "Worker initialization failed")),
            );
            return;
          }
        }
        // Track progress for model file downloads
        const msgWithFile = msg as ProgressInfo & { file?: string };
        if (msgWithFile.file) trackProgress(msg as ProgressInfo);
      };

      worker.addEventListener("message", onMessage);
      worker.postMessage({
        type: "load",
        data: {
          modelId: this.modelId,
          dtype: this.config.dtype,
          device: this.config.device,
          maxNewTokens: this.config.maxNewTokens,
        },
      });
    });
  }
}
