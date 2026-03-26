import {
  EmbeddingModelV3,
  NoSuchModelError,
  ProviderV3,
} from "@ai-sdk/provider";
import {
  WebLLMLanguageModel,
  WebLLMModelId,
  WebLLMSettings,
} from "./chat/web-llm-language-model";
import {
  WebLLMEmbeddingModel,
  WebLLMEmbeddingModelId,
  WebLLMEmbeddingSettings,
} from "./embedding/web-llm-embedding-model";

/**
 * Per-call provider options for WebLLM, passed via `providerOptions["web-llm"]`
 * in each `generateText` / `streamText` call.
 *
 * Distinct from {@link WebLLMSettings}, which are set once at model-creation time.
 */
export interface WebLLMCallProviderOptions {
  /**
   * Replaces the default tool-calling prompt text that appears before the
   * generated tool schemas JSON block.
   *
   * If this or `afterToolSchemasPrompt` is provided, the default tool-use
   * prompt is not emitted. An empty string is treated as absent.
   */
  beforeToolSchemasPrompt?: string;
  /**
   * Replaces the default tool-calling prompt text that appears after the
   * generated tool schemas JSON block.
   *
   * If this or `beforeToolSchemasPrompt` is provided, the default tool-use
   * prompt is not emitted. An empty string is treated as absent.
   */
  afterToolSchemasPrompt?: string;
  /**
   * Additional generation config passed directly to the WebLLM engine.
   * @see https://webllm.mlc.ai/docs/user/api_reference.html#generationconfig
   */
  extra_body?: {
    enable_thinking?: boolean;
    enable_latency_breakdown?: boolean;
  };
}

export interface WebLLMProvider extends ProviderV3 {
  (modelId: WebLLMModelId, settings?: WebLLMSettings): WebLLMLanguageModel;

  /**
   * Creates a model for text generation.
   */
  languageModel(
    modelId: WebLLMModelId,
    settings?: WebLLMSettings,
  ): WebLLMLanguageModel;

  /**
   * Creates a model for text generation.
   */
  chat(modelId: WebLLMModelId, settings?: WebLLMSettings): WebLLMLanguageModel;

  /**
   * Creates a model for text embeddings.
   */
  embedding(
    modelId: WebLLMEmbeddingModelId,
    settings?: WebLLMEmbeddingSettings,
  ): EmbeddingModelV3;

  /**
   * Creates a model for text embeddings.
   */
  embeddingModel: (
    modelId: WebLLMEmbeddingModelId,
    settings?: WebLLMEmbeddingSettings,
  ) => EmbeddingModelV3;
}

/**
 * Create a WebLLM provider instance.
 */
export function createWebLLM(): WebLLMProvider {
  const createLanguageModel = (
    modelId: WebLLMModelId,
    settings?: WebLLMSettings,
  ) => {
    return new WebLLMLanguageModel(modelId, settings);
  };

  const createEmbeddingModel = (
    modelId: WebLLMEmbeddingModelId,
    settings?: WebLLMEmbeddingSettings,
  ) => {
    return new WebLLMEmbeddingModel(modelId, settings);
  };

  const provider = function (
    modelId: WebLLMModelId,
    settings?: WebLLMSettings,
  ) {
    if (new.target) {
      throw new Error(
        "The WebLLM model function cannot be called with the new keyword.",
      );
    }

    return createLanguageModel(modelId, settings);
  };

  provider.specificationVersion = "v3" as const;
  provider.languageModel = createLanguageModel;
  provider.chat = createLanguageModel;
  provider.embedding = createEmbeddingModel;
  provider.embeddingModel = createEmbeddingModel;

  provider.imageModel = (modelId: string) => {
    throw new NoSuchModelError({ modelId, modelType: "imageModel" });
  };

  provider.speechModel = (modelId: string) => {
    throw new NoSuchModelError({ modelId, modelType: "speechModel" });
  };

  provider.transcriptionModel = (modelId: string) => {
    throw new NoSuchModelError({ modelId, modelType: "transcriptionModel" });
  };

  return provider as WebLLMProvider;
}

/**
 * Default WebLLM provider instance
 *
 * @example
 * ```typescript
 * import { webLLM } from "@browser-ai/web-llm";
 *
 * // Language model
 * const chat = webLLM("Llama-3.2-3B-Instruct-q4f16_1-MLC");
 *
 * // Embedding model
 * const embed = webLLM.embeddingModel("snowflake-arctic-embed-m-q0f32-MLC-b32");
 * ```
 */
export const webLLM = createWebLLM();
