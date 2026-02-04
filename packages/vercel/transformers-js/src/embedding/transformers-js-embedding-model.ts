import {
  EmbeddingModelV3,
  TooManyEmbeddingValuesForCallError,
  LoadSettingError,
  EmbeddingModelV3CallOptions,
  EmbeddingModelV3Result,
} from "@ai-sdk/provider";
import {
  pipeline,
  AutoTokenizer,
  type PreTrainedTokenizer,
  type ProgressInfo,
  type PretrainedModelOptions,
  type FeatureExtractionPipeline,
  type Tensor,
} from "@huggingface/transformers";
import type { DownloadProgressCallback } from "@browser-ai/shared";

export type TransformersJSEmbeddingModelId = string;

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

export interface TransformersJSEmbeddingSettings extends Pick<
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
   * Whether to normalize embeddings
   * @default true
   */
  normalize?: boolean;
  /**
   * Pooling strategy for token embeddings
   * @default "mean"
   */
  pooling?: "mean" | "cls" | "max";
  /**
   * Maximum number of tokens per input
   * @default 512
   */
  maxTokens?: number;
}

export class TransformersJSEmbeddingModel implements EmbeddingModelV3 {
  readonly specificationVersion = "v3";
  readonly provider = "transformers-js";
  readonly modelId: TransformersJSEmbeddingModelId;
  readonly maxEmbeddingsPerCall = 100; // Reasonable limit for browser
  readonly supportsParallelCalls = false;

  private readonly config: TransformersJSEmbeddingSettings & {
    modelId: TransformersJSEmbeddingModelId;
  };
  private pipeline: FeatureExtractionPipeline | null = null;
  private tokenizer: PreTrainedTokenizer | null = null;
  private isInitialized = false;
  private initializationPromise?: Promise<void>;

  constructor(
    modelId: TransformersJSEmbeddingModelId,
    options: TransformersJSEmbeddingSettings = {},
  ) {
    this.modelId = modelId;
    this.config = {
      modelId,
      device: "auto",
      dtype: "auto",
      normalize: true,
      pooling: "mean",
      maxTokens: 512,
      ...options,
    };
  }

  private async getSession(
    onInitProgress?: DownloadProgressCallback,
  ): Promise<[PreTrainedTokenizer, FeatureExtractionPipeline]> {
    if (this.pipeline && this.tokenizer && this.isInitialized) {
      return [this.tokenizer, this.pipeline];
    }

    if (this.initializationPromise) {
      await this.initializationPromise;
      if (this.pipeline && this.tokenizer) {
        return [this.tokenizer, this.pipeline];
      }
    }

    this.initializationPromise = this._initializeModel(onInitProgress);
    await this.initializationPromise;

    if (!this.pipeline || !this.tokenizer) {
      throw new LoadSettingError({
        message: "Embedding model initialization failed",
      });
    }

    return [this.tokenizer, this.pipeline];
  }

  private async _initializeModel(
    onInitProgress?: DownloadProgressCallback,
  ): Promise<void> {
    try {
      const { device, dtype } = this.config;
      const progress_callback = this.createProgressTracker(onInitProgress);

      // Set device based on environment
      const resolvedDevice = this.resolveDevice(
        device as string,
      ) as PretrainedModelOptions["device"];
      const resolvedDtype = this.resolveDtype(
        dtype as string,
      ) as PretrainedModelOptions["dtype"];

      // Create tokenizer and pipeline
      const [tokenizer, embeddingPipeline] = await Promise.all([
        AutoTokenizer.from_pretrained(this.modelId, {
          legacy: true,
          progress_callback,
        }),
        pipeline("feature-extraction", this.modelId, {
          device: resolvedDevice,
          dtype: resolvedDtype,
          progress_callback,
        }),
      ]);

      this.tokenizer = tokenizer;
      this.pipeline = embeddingPipeline as FeatureExtractionPipeline;

      onInitProgress?.(1.0);
      this.isInitialized = true;
    } catch (error) {
      this.pipeline = null;
      this.tokenizer = null;
      this.isInitialized = false;
      this.initializationPromise = undefined;

      throw new LoadSettingError({
        message: `Failed to initialize TransformersJS embedding model: ${error instanceof Error ? error.message : "Unknown error"}`,
      });
    }
  }

  private resolveDevice(device?: string): string {
    if (device && device !== "auto") {
      return device;
    }

    if (isServerEnvironment()) {
      // In server environment, prefer CPU unless explicitly set
      return "cpu";
    }

    // In browser environment, auto-detect WebGPU support
    if (
      isBrowserEnvironment() &&
      typeof navigator !== "undefined" &&
      (navigator as any).gpu
    ) {
      return "webgpu";
    }

    return "cpu";
  }

  private resolveDtype(dtype?: string): string {
    if (dtype && dtype !== "auto") {
      return dtype;
    }

    return "q8";
  }

  private createProgressTracker(onInitProgress?: DownloadProgressCallback) {
    const fileProgress = new Map<string, { loaded: number; total: number }>();

    return (p: ProgressInfo) => {
      // Pass through raw progress
      this.config.rawInitProgressCallback?.(p);

      if (!onInitProgress) return;

      // Type guard to check if p has file property
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

  private applyPooling(embeddings: number[][], pooling: string): number[] {
    if (pooling === "cls") {
      return embeddings[0]; // Return first token (CLS token)
    }

    const hiddenSize = embeddings[0].length;

    if (pooling === "max") {
      const pooled = new Array(hiddenSize).fill(-Infinity);
      for (const embedding of embeddings) {
        for (let j = 0; j < hiddenSize; j++) {
          pooled[j] = Math.max(pooled[j], embedding[j]);
        }
      }
      return pooled;
    }

    // Default: mean pooling
    const result = new Array(hiddenSize).fill(0);
    for (const embedding of embeddings) {
      for (let j = 0; j < hiddenSize; j++) {
        result[j] += embedding[j];
      }
    }
    return result.map((val) => val / embeddings.length);
  }

  private normalizeVector(vector: number[]): number[] {
    const norm = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    return norm > 0 ? vector.map((val) => val / norm) : vector;
  }

  /**
   * Check the availability of the TransformersJS embedding model
   */
  public async availability(): Promise<
    "unavailable" | "downloadable" | "available"
  > {
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
  ): Promise<TransformersJSEmbeddingModel> {
    await this._initializeModel(onDownloadProgress);
    return this;
  }

  async doEmbed(
    options: EmbeddingModelV3CallOptions,
  ): Promise<EmbeddingModelV3Result> {
    const { values } = options;

    if (values.length > this.maxEmbeddingsPerCall) {
      throw new TooManyEmbeddingValuesForCallError({
        provider: this.provider,
        modelId: this.modelId,
        maxEmbeddingsPerCall: this.maxEmbeddingsPerCall,
        values: values,
      });
    }

    const [tokenizer, model] = await this.getSession(
      this.config.initProgressCallback,
    );

    const embeddings = await Promise.all(
      values.map(async (text) => {
        try {
          // Tokenize the input
          const tokens = await tokenizer(text, {
            padding: true,
            truncation: true,
            max_length: this.config.maxTokens,
            return_tensors: false,
          });

          // Get embeddings
          const result = await model(text, {
            pooling: "none", // We'll handle pooling ourselves
            normalize: false, // We'll handle normalization ourselves
          });

          let embedding: number[];

          // Handle Tensor result from transformers.js
          if (
            result &&
            typeof result === "object" &&
            "data" in result &&
            "dims" in result
          ) {
            // Result is a Tensor from transformers.js
            const tensor = result as Tensor;
            const data = Array.from(tensor.data) as number[];
            const dims = tensor.dims as number[];

            if (dims.length === 3) {
              // [batch_size, sequence_length, hidden_size] - needs pooling
              const [batchSize, seqLength, hiddenSize] = dims;
              const sequences: number[][] = [];

              // Reshape flat array back to [sequence_length, hidden_size] for first batch
              for (let i = 0; i < seqLength; i++) {
                const sequence: number[] = [];
                for (let j = 0; j < hiddenSize; j++) {
                  sequence.push(data[i * hiddenSize + j]);
                }
                sequences.push(sequence);
              }

              embedding = this.applyPooling(
                sequences,
                this.config.pooling || "mean",
              );
            } else if (dims.length === 2) {
              // [sequence_length, hidden_size] - needs pooling
              const [seqLength, hiddenSize] = dims;
              const sequences: number[][] = [];

              for (let i = 0; i < seqLength; i++) {
                const sequence: number[] = [];
                for (let j = 0; j < hiddenSize; j++) {
                  sequence.push(data[i * hiddenSize + j]);
                }
                sequences.push(sequence);
              }

              embedding = this.applyPooling(
                sequences,
                this.config.pooling || "mean",
              );
            } else if (dims.length === 1) {
              // Already pooled [hidden_size]
              embedding = data;
            } else {
              throw new Error(`Unsupported tensor dimensions: ${dims}`);
            }
          } else if (
            Array.isArray(result) &&
            Array.isArray(result[0]) &&
            Array.isArray(result[0][0])
          ) {
            // Result is [batch_size, sequence_length, hidden_size]
            embedding = this.applyPooling(
              result[0] as number[][],
              this.config.pooling || "mean",
            );
          } else if (Array.isArray(result) && typeof result[0] === "number") {
            // Result is already pooled
            embedding = result as number[];
          } else {
            console.error("Unexpected result format:", result);
            throw new Error("Unexpected embedding result format");
          }

          // Normalize if requested
          if (this.config.normalize) {
            embedding = this.normalizeVector(embedding);
          }

          return {
            embedding,
            tokenCount: Array.isArray(tokens.input_ids)
              ? tokens.input_ids.length
              : 0,
          };
        } catch (error) {
          throw new Error(`Failed to generate embedding for text: ${error}`);
        }
      }),
    );

    const totalTokens = embeddings.reduce(
      (sum, { tokenCount }) => sum + tokenCount,
      0,
    );

    return {
      embeddings: embeddings.map(({ embedding }) => embedding),
      usage: { tokens: totalTokens },
      warnings: [],
    };
  }
}
