import {
  EmbeddingModelV3,
  EmbeddingModelV3CallOptions,
  EmbeddingModelV3Result,
  TooManyEmbeddingValuesForCallError,
  LoadSettingError,
} from "@ai-sdk/provider";
import {
  AppConfig,
  CreateWebWorkerMLCEngine,
  InitProgressReport,
  MLCEngine,
  MLCEngineConfig,
  MLCEngineInterface,
} from "@mlc-ai/web-llm";
import { Availability } from "./types";
import { isMobile, checkWebGPU } from "./utils/browser";

export type WebLLMEmbeddingModelId = string;

export interface WebLLMEmbeddingSettings {
  /**
   * Custom app configuration for WebLLM
   */
  appConfig?: AppConfig;
  /**
   * Progress callback for model initialization
   */
  initProgressCallback?: (progress: InitProgressReport) => void;
  /**
   * Engine configuration options
   */
  engineConfig?: MLCEngineConfig;
  /**
   * A web worker instance to run the model in.
   * When provided, the model will run in a separate thread.
   *
   * @default undefined
   */
  worker?: Worker;
  /**
   * Maximum number of texts to embed in a single call.
   * @default 100
   */
  maxEmbeddingsPerCall?: number;
}

type WebLLMEmbeddingConfig = {
  provider: string;
  modelId: WebLLMEmbeddingModelId;
  options: WebLLMEmbeddingSettings;
};

export class WebLLMEmbeddingModel implements EmbeddingModelV3 {
  readonly specificationVersion = "v3";
  readonly provider = "web-llm";
  readonly modelId: WebLLMEmbeddingModelId;
  readonly maxEmbeddingsPerCall: number;
  readonly supportsParallelCalls = false;

  private readonly config: WebLLMEmbeddingConfig;
  private engine?: MLCEngineInterface;
  private isInitialized = false;
  private initializationPromise?: Promise<void>;

  constructor(
    modelId: WebLLMEmbeddingModelId,
    options: WebLLMEmbeddingSettings = {},
  ) {
    this.modelId = modelId;
    this.maxEmbeddingsPerCall = options.maxEmbeddingsPerCall ?? 100;
    this.config = {
      provider: this.provider,
      modelId,
      options,
    };
  }

  /**
   * Check if the model is initialized and ready to use
   */
  get isModelInitialized(): boolean {
    return this.isInitialized;
  }

  private async getEngine(
    options?: MLCEngineConfig,
    onInitProgress?: (progress: InitProgressReport) => void,
  ): Promise<MLCEngineInterface> {
    const availability = await this.availability();
    if (availability === "unavailable") {
      throw new LoadSettingError({
        message:
          "WebLLM is not available. This library requires a browser with WebGPU support.",
      });
    }

    if (this.engine && this.isInitialized) return this.engine;

    if (this.initializationPromise) {
      await this.initializationPromise;
      if (this.engine) return this.engine;
    }

    this.initializationPromise = this._initializeEngine(
      options,
      onInitProgress,
    );
    await this.initializationPromise;

    if (!this.engine) {
      throw new LoadSettingError({
        message: "Engine initialization failed",
      });
    }

    return this.engine;
  }

  private async _initializeEngine(
    options?: MLCEngineConfig,
    onInitProgress?: (progress: InitProgressReport) => void,
  ): Promise<void> {
    try {
      const engineConfig = {
        ...this.config.options.engineConfig,
        ...options,
        initProgressCallback:
          onInitProgress || this.config.options.initProgressCallback,
      };

      if (this.config.options.worker) {
        this.engine = await CreateWebWorkerMLCEngine(
          this.config.options.worker,
          this.modelId,
          engineConfig,
        );
      } else {
        this.engine = new MLCEngine(engineConfig);
        await this.engine.reload(this.modelId);
      }

      this.isInitialized = true;
    } catch (error) {
      this.engine = undefined;
      this.isInitialized = false;
      this.initializationPromise = undefined;

      throw new LoadSettingError({
        message: `Failed to initialize WebLLM embedding engine: ${error instanceof Error ? error.message : "Unknown error"}`,
      });
    }
  }

  /**
   * Check the availability of the WebLLM embedding model
   * @returns Promise resolving to "unavailable", "available", or "downloadable"
   */
  public async availability(): Promise<Availability> {
    if (this.isInitialized) {
      return "available";
    }

    if (this.config.options.worker && isMobile()) {
      return "downloadable";
    }

    const supported = checkWebGPU();
    return supported ? "downloadable" : "unavailable";
  }

  /**
   * Creates an engine session with download progress monitoring.
   *
   * @example
   * ```typescript
   * const engine = await model.createSessionWithProgress(
   *   (progress) => {
   *     console.log(`Download progress: ${Math.round(progress.progress * 100)}%`);
   *   }
   * );
   * ```
   *
   * @param onInitProgress Optional callback receiving progress reports during model download
   * @returns Promise resolving to a configured WebLLM engine
   * @throws {LoadSettingError} When WebLLM isn't available or model is unavailable
   */
  public async createSessionWithProgress(
    onInitProgress?: (progress: InitProgressReport) => void,
  ): Promise<MLCEngineInterface> {
    return this.getEngine(undefined, onInitProgress);
  }

  /**
   * Embed texts using the WebLLM embedding model
   */
  public async doEmbed(
    options: EmbeddingModelV3CallOptions,
  ): Promise<EmbeddingModelV3Result> {
    const { values, abortSignal } = options;

    if (values.length > this.maxEmbeddingsPerCall) {
      throw new TooManyEmbeddingValuesForCallError({
        provider: this.provider,
        modelId: this.modelId,
        maxEmbeddingsPerCall: this.maxEmbeddingsPerCall,
        values,
      });
    }

    if (abortSignal?.aborted) {
      throw new Error("Operation was aborted");
    }

    const engine = await this.getEngine();

    const abortHandler = () => {
      engine.interruptGenerate();
    };

    if (abortSignal) {
      abortSignal.addEventListener("abort", abortHandler);
    }

    try {
      const response = await engine.embeddings.create({
        input: values,
        model: this.modelId,
        ...(abortSignal &&
          !this.config.options.worker && { signal: abortSignal }),
      });

      const sortedEmbeddings = response.data
        .sort((a, b) => a.index - b.index)
        .map((e) => e.embedding);

      return {
        embeddings: sortedEmbeddings,
        usage: {
          tokens: response.usage.total_tokens,
        },
        providerMetadata: {
          webllm: {
            model: response.model,
            promptTokens: response.usage.prompt_tokens,
            totalTokens: response.usage.total_tokens,
            prefillTokensPerSecond: response.usage.extra?.prefill_tokens_per_s,
          },
        },
        warnings: [],
      };
    } catch (error) {
      throw new Error(
        `WebLLM embedding failed: ${error instanceof Error ? error.message : "Unknown error"}`,
      );
    } finally {
      if (abortSignal) {
        abortSignal.removeEventListener("abort", abortHandler);
      }
    }
  }
}
