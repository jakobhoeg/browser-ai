import {
  AutoTokenizer,
  AutoProcessor,
  WhisperForConditionalGeneration,
  TextStreamer,
  full,
  type ProgressInfo,
  type PreTrainedTokenizer,
  type Processor,
  type PreTrainedModel,
} from "@huggingface/transformers";

export interface TranscriptionWorkerMessage {
  type: "load" | "generate" | "interrupt" | "reset";
  data?: any;
}

export interface TranscriptionWorkerResponse {
  status: "loading" | "ready" | "start" | "update" | "complete" | "error";
  output?: string | string[];
  data?: string;
  tps?: number;
  numTokens?: number;
}

export interface TranscriptionWorkerLoadOptions {
  modelId?: string;
  dtype?: string;
  device?: string;
}

export interface TranscriptionWorkerGlobalScope {
  postMessage(message: any): void;
  addEventListener(type: string, listener: (e: any) => void): void;
}

declare const self: TranscriptionWorkerGlobalScope;

type TranscriptionModelInstance = [
  PreTrainedTokenizer,
  Processor,
  PreTrainedModel,
];

class TranscriptionModelManager {
  private static configs = new Map<string, TranscriptionWorkerLoadOptions>();
  private static instances = new Map<
    string,
    Promise<TranscriptionModelInstance>
  >();

  static configure(key: string, options: TranscriptionWorkerLoadOptions) {
    this.configs.set(key, options);
  }

  static async getInstance(
    key: string,
    progressCallback?: (progress: ProgressInfo) => void,
  ): Promise<TranscriptionModelInstance> {
    const cached = this.instances.get(key);
    if (cached) return cached;

    const config = this.configs.get(key);
    if (!config || !config.modelId) {
      throw new Error(`No configuration found for key: ${key}`);
    }

    const { modelId, dtype = "auto", device = "auto" } = config;

    const instancePromise = this.createTranscriptionModel(modelId, {
      dtype,
      device,
      progressCallback,
    });

    this.instances.set(key, instancePromise);
    return instancePromise;
  }

  private static async createTranscriptionModel(
    modelId: string,
    options: any,
  ): Promise<TranscriptionModelInstance> {
    const [tokenizer, processor, model] = await Promise.all([
      AutoTokenizer.from_pretrained(modelId, {
        progress_callback: options.progressCallback,
      }),
      AutoProcessor.from_pretrained(modelId, {
        progress_callback: options.progressCallback,
      }),
      WhisperForConditionalGeneration.from_pretrained(modelId, {
        dtype: options.dtype || {
          encoder_model: "fp32",
          decoder_model_merged: "q4",
        },
        device: options.device || "auto",
        progress_callback: options.progressCallback,
      }),
    ]);
    return [tokenizer, processor, model];
  }

  static clearCache() {
    this.instances.clear();
  }
}

/**
 * Worker handler for TransformersJS transcription models that runs in a Web Worker context.
 *
 * This class manages the lifecycle of transcription models in a worker thread, providing
 * audio transcription capabilities without blocking the main UI thread. It handles model
 * loading, initialization, transcription generation, and communication with the main thread.
 *
 * @example
 * ```typescript
 * // worker.ts
 * import { TransformersJSTranscriptionWorkerHandler } from "@browser-ai/transformers-js";
 *
 * const handler = new TransformersJSTranscriptionWorkerHandler();
 * self.onmessage = (msg: MessageEvent) => {
 * handler.onmessage(msg);
 * };
 * ```
 */
export class TransformersJSTranscriptionWorkerHandler {
  private processing = false;
  private currentModelKey = "default";

  async generate({
    audio,
    language,
    maxNewTokens,
  }: {
    audio: any;
    language?: string;
    maxNewTokens?: number;
  }) {
    if (this.processing) return;
    this.processing = true;

    try {
      // Tell the main thread we are starting
      this.sendMessage({ status: "start" });

      // Retrieve the transcription model
      const [tokenizer, processor, model] =
        await TranscriptionModelManager.getInstance(this.currentModelKey);

      // Setup performance tracking
      let startTime: number | undefined;
      let numTokens = 0;
      const token_callback_function = () => {
        startTime ??= performance.now();
        if (numTokens++ > 0) {
          const tps = (numTokens / (performance.now() - startTime!)) * 1000;
        }
      };

      const callback_function = (output: string) => {
        const tps = startTime
          ? (numTokens / (performance.now() - startTime)) * 1000
          : undefined;
        this.sendUpdate(output, tps, numTokens);
      };

      const streamer = new TextStreamer(tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function,
        token_callback_function,
      });

      // Convert audio data - expecting Float32Array from properly decoded audio
      let audioFloat32: Float32Array;
      if (Array.isArray(audio)) {
        audioFloat32 = new Float32Array(audio);
      } else if (audio instanceof Float32Array) {
        audioFloat32 = audio;
      } else {
        // Convert other formats to Float32Array
        audioFloat32 = new Float32Array(audio);
      }

      // Process the audio
      const inputs = await processor(audioFloat32);

      // Generate transcription with proper error handling
      const outputs = await (model as any).generate({
        ...inputs,
        max_new_tokens: maxNewTokens || 448,
        language,
        streamer,
      });

      const decoded = tokenizer.batch_decode(outputs, {
        skip_special_tokens: true,
      });

      // Send the output back to the main thread
      this.sendMessage({
        status: "complete",
        output: decoded,
      });
    } catch (error) {
      this.sendError(error instanceof Error ? error.message : String(error));
    } finally {
      this.processing = false;
    }
  }

  async load(options?: TranscriptionWorkerLoadOptions) {
    try {
      TranscriptionModelManager.clearCache();

      const modelId = options?.modelId;

      TranscriptionModelManager.configure(this.currentModelKey, {
        ...options,
        modelId,
      });

      this.sendMessage({ status: "loading", data: "Loading model..." });

      const throttledProgress = this.createThrottledProgressCallback();

      const [tokenizer, processor, model] =
        await TranscriptionModelManager.getInstance(
          this.currentModelKey,
          throttledProgress,
        );

      this.sendMessage({
        status: "loading",
        data: "Compiling shaders and warming up model...",
      });

      // Run model with dummy input to compile shaders
      try {
        const numMelBins = (model.config as any).num_mel_bins ?? 80;
        await (model as any).generate({
          inputs: full([1, numMelBins, 3000], 0.0),
          max_new_tokens: 1,
        });
      } catch (error) {
        // Ignore warmup errors
        console.warn("Model warmup failed:", error);
      }

      this.sendMessage({ status: "ready" });
    } catch (error) {
      console.error("Error in transcription worker load:", error);
      this.sendError(error instanceof Error ? error.message : String(error));
    }
  }

  interrupt() {
    // For transcription, we don't have the same stopping criteria as text generation
    // but we can set a flag to prevent new generations
    this.processing = false;
  }

  reset() {
    this.processing = false;
    TranscriptionModelManager.clearCache();
  }

  private sendMessage(message: {
    status: "loading" | "ready" | "start" | "complete";
    data?: string;
    output?: string | string[];
  }) {
    self.postMessage(message);
  }

  private sendUpdate(output: string, tps?: number, numTokens?: number) {
    self.postMessage({ status: "update", output, tps, numTokens });
  }

  private sendError(message: string) {
    self.postMessage({ status: "error", data: message });
  }

  private createThrottledProgressCallback() {
    const throttleMs = 100;
    let lastProgressTs = 0;

    return (progress: ProgressInfo) => {
      const now = performance?.now?.() ?? Date.now();
      if (progress.status === "progress") {
        if (now - lastProgressTs < throttleMs) return;
        lastProgressTs = now;
      }
      self.postMessage(progress);
    };
  }

  onmessage(e: MessageEvent<TranscriptionWorkerMessage>) {
    try {
      const { type, data } = e.data || ({} as TranscriptionWorkerMessage);
      switch (type) {
        case "load":
          this.load(data);
          break;
        case "generate":
          this.generate(data);
          break;
        case "interrupt":
          this.interrupt();
          break;
        case "reset":
          this.reset();
          break;
        default:
          this.sendError(`Unknown message type: ${type}`);
          break;
      }
    } catch (error) {
      this.sendError(error instanceof Error ? error.message : String(error));
    }
  }
}
