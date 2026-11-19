import { NoSuchModelError, ProviderV4 } from "@ai-sdk/provider";
import {
  WllamaLanguageModel,
  WllamaModelId,
  WllamaSettings,
} from "./chat/wllama-language-model";

export interface WllamaProvider extends ProviderV4 {
  (modelId: WllamaModelId, settings?: WllamaSettings): WllamaLanguageModel;

  /**
   * Creates a model for text generation.
   */
  languageModel(
    modelId: WllamaModelId,
    settings?: WllamaSettings,
  ): WllamaLanguageModel;

  /**
   * Creates a model for text generation.
   */
  chat(modelId: WllamaModelId, settings?: WllamaSettings): WllamaLanguageModel;
}

/**
 * Create a wllama provider instance.
 */
export function createWllama(): WllamaProvider {
  const createLanguageModel = (
    modelId: WllamaModelId,
    settings?: WllamaSettings,
  ) => {
    return new WllamaLanguageModel(modelId, settings);
  };

  const provider = function (
    modelId: WllamaModelId,
    settings?: WllamaSettings,
  ) {
    if (new.target) {
      throw new Error(
        "The wllama model function cannot be called with the new keyword.",
      );
    }

    return createLanguageModel(modelId, settings);
  };

  provider.specificationVersion = "v4" as const;
  provider.languageModel = createLanguageModel;
  provider.chat = createLanguageModel;

  provider.embeddingModel = (modelId: string) => {
    throw new NoSuchModelError({ modelId, modelType: "embeddingModel" });
  };

  provider.imageModel = (modelId: string) => {
    throw new NoSuchModelError({ modelId, modelType: "imageModel" });
  };

  provider.speechModel = (modelId: string) => {
    throw new NoSuchModelError({ modelId, modelType: "speechModel" });
  };

  provider.transcriptionModel = (modelId: string) => {
    throw new NoSuchModelError({ modelId, modelType: "transcriptionModel" });
  };

  return provider as WllamaProvider;
}

/**
 * Default wllama provider instance
 *
 * @example
 * ```typescript
 * import { wllama } from "@browser-ai/wllama";
 *
 * const chat = wllama("hf:ggml-org/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf");
 * ```
 */
export const wllama = createWllama();
