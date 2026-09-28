import {
  EmbeddingModelV4,
  NoSuchModelError,
  ProviderV4,
} from "@ai-sdk/provider";
import {
  LiteRTLMLanguageModel,
  type LiteRTLMModelId,
  type LiteRTLMSettings,
} from "./chat/litert-lm-language-model";

export interface LiteRTLMProvider extends ProviderV4 {
  (
    modelId: LiteRTLMModelId,
    settings?: LiteRTLMSettings,
  ): LiteRTLMLanguageModel;

  /**
   * Creates a model for text generation.
   */
  languageModel(
    modelId: LiteRTLMModelId,
    settings?: LiteRTLMSettings,
  ): LiteRTLMLanguageModel;

  /**
   * Creates a model for text generation.
   */
  chat(
    modelId: LiteRTLMModelId,
    settings?: LiteRTLMSettings,
  ): LiteRTLMLanguageModel;
}

/**
 * Create a LiteRT-LM provider instance.
 */
export function createLiteRTLM(): LiteRTLMProvider {
  const createLanguageModel = (
    modelId: LiteRTLMModelId,
    settings?: LiteRTLMSettings,
  ) => {
    return new LiteRTLMLanguageModel(modelId, settings);
  };

  const provider = function (
    modelId: LiteRTLMModelId,
    settings?: LiteRTLMSettings,
  ) {
    if (new.target) {
      throw new Error(
        "The LiteRT-LM model function cannot be called with the new keyword.",
      );
    }

    return createLanguageModel(modelId, settings);
  };

  provider.specificationVersion = "v4" as const;
  provider.languageModel = createLanguageModel;
  provider.chat = createLanguageModel;

  // The LiteRT-LM Web SDK is currently text-in / text-out, so these model
  // types are not supported. Throw a clear error to match the other providers.
  provider.embedding = (modelId: string): EmbeddingModelV4 => {
    throw new NoSuchModelError({ modelId, modelType: "embeddingModel" });
  };
  provider.embeddingModel = provider.embedding;

  provider.imageModel = (modelId: string) => {
    throw new NoSuchModelError({ modelId, modelType: "imageModel" });
  };

  provider.speechModel = (modelId: string) => {
    throw new NoSuchModelError({ modelId, modelType: "speechModel" });
  };

  provider.transcriptionModel = (modelId: string) => {
    throw new NoSuchModelError({ modelId, modelType: "transcriptionModel" });
  };

  return provider as LiteRTLMProvider;
}

/**
 * Default LiteRT-LM provider instance.
 *
 * @example
 * ```typescript
 * import { liteRTLM, GEMMA_4_E2B_WEB } from "@browser-ai/litert-lm";
 * import { streamText } from "ai";
 *
 * const result = streamText({
 *   model: liteRTLM(GEMMA_4_E2B_WEB),
 *   prompt: "Explain WebGPU in one sentence.",
 * });
 * ```
 */
export const liteRTLM = createLiteRTLM();
