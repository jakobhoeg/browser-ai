import {
  EmbeddingModelV3,
  NoSuchModelError,
  ProviderV3,
} from "@ai-sdk/provider";
import {
  BrowserAIChatLanguageModel,
  BrowserAIChatModelId,
  BrowserAIChatSettings,
} from "./chat/browser-ai-language-model";
import {
  BrowserAIEmbeddingModel,
  BrowserAIEmbeddingModelSettings,
} from "./embedding/browser-ai-embedding-model";

export interface BrowserAIProvider extends ProviderV3 {
  (
    modelId?: BrowserAIChatModelId,
    settings?: BrowserAIChatSettings,
  ): BrowserAIChatLanguageModel;

  /**
   * Creates a model for text generation.
   */
  languageModel(
    modelId: BrowserAIChatModelId,
    settings?: BrowserAIChatSettings,
  ): BrowserAIChatLanguageModel;

  /**
   * Creates a model for text generation.
   */
  chat(
    modelId: BrowserAIChatModelId,
    settings?: BrowserAIChatSettings,
  ): BrowserAIChatLanguageModel;

  embedding(
    modelId: "embedding",
    settings?: BrowserAIEmbeddingModelSettings,
  ): EmbeddingModelV3;

  embeddingModel: (
    modelId: "embedding",
    settings?: BrowserAIEmbeddingModelSettings,
  ) => EmbeddingModelV3;

  // Not implemented
  imageModel(modelId: string): never;
  speechModel(modelId: string): never;
  transcriptionModel(modelId: string): never;
}

export interface BrowserAIProviderSettings {
  // Currently empty - provider settings are minimal for BrowserAI
  // Future provider-level settings can be added here
}

/**
 * Create a BrowserAI provider instance.
 */
export function createBrowserAI(
  options: BrowserAIProviderSettings = {},
): BrowserAIProvider {
  const createChatModel = (
    modelId: BrowserAIChatModelId,
    settings?: BrowserAIChatSettings,
  ) => {
    return new BrowserAIChatLanguageModel(modelId, settings);
  };

  const createEmbeddingModel = (
    modelId: "embedding",
    settings?: BrowserAIEmbeddingModelSettings,
  ) => {
    return new BrowserAIEmbeddingModel(settings);
  };

  const provider = function (
    modelId: BrowserAIChatModelId = "text",
    settings?: BrowserAIChatSettings,
  ) {
    if (new.target) {
      throw new Error(
        "The BrowserAI model function cannot be called with the new keyword.",
      );
    }

    return createChatModel(modelId, settings);
  };

  provider.specificationVersion = "v3" as const;
  provider.languageModel = createChatModel;
  provider.chat = createChatModel;
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

  return provider;
}

/**
 * Default BrowserAI provider instance.
 */
export const browserAI = createBrowserAI();
