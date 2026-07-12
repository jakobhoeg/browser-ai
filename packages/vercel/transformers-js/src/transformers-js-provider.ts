import {
  EmbeddingModelV4,
  LanguageModelV4,
  NoSuchModelError,
  ProviderV4,
  TranscriptionModelV4,
} from "@ai-sdk/provider";
import {
  TransformersJSLanguageModel,
  TransformersJSModelId,
  TransformersJSModelSettings,
  isServerEnvironment,
} from "./chat/transformers-js-language-model";
import {
  TransformersJSEmbeddingModel,
  TransformersJSEmbeddingModelId,
  TransformersJSEmbeddingSettings,
} from "./embedding/transformers-js-embedding-model";
import {
  TransformersJSTranscriptionModel,
  TransformersJSTranscriptionModelId,
  TransformersJSTranscriptionSettings,
} from "./transcription/transformers-js-transcription-model";

export interface TransformersJSProvider extends ProviderV4 {
  (
    modelId: TransformersJSModelId,
    settings?: TransformersJSModelSettings,
  ): TransformersJSLanguageModel;

  /**
   * Creates a model for text generation.
   */
  languageModel(
    modelId: TransformersJSModelId,
    settings?: TransformersJSModelSettings,
  ): TransformersJSLanguageModel;

  /**
   * Creates a model for text generation.
   */
  chat(
    modelId: TransformersJSModelId,
    settings?: TransformersJSModelSettings,
  ): TransformersJSLanguageModel;

  embedding(
    modelId: TransformersJSEmbeddingModelId,
    settings?: TransformersJSEmbeddingSettings,
  ): EmbeddingModelV4;

  embeddingModel: (
    modelId: TransformersJSEmbeddingModelId,
    settings?: TransformersJSEmbeddingSettings,
  ) => EmbeddingModelV4;

  transcription(
    modelId: TransformersJSTranscriptionModelId,
    settings?: TransformersJSTranscriptionSettings,
  ): TranscriptionModelV4;

  transcriptionModel: (
    modelId: TransformersJSTranscriptionModelId,
    settings?: TransformersJSTranscriptionSettings,
  ) => TranscriptionModelV4;
}

export interface TransformersJSProviderSettings {
  // Currently empty - provider settings are minimal for TransformersJS
  // Future provider-level settings can be added here
}

/**
 * Create a TransformersJS provider instance.
 */
export function createTransformersJS(
  options: TransformersJSProviderSettings = {},
): TransformersJSProvider {
  const createChatModel = (
    modelId: TransformersJSModelId,
    settings?: TransformersJSModelSettings,
  ) => {
    // On the server, return a singleton per model + device + dtype + isVision configuration
    // so initialization state persists across uses (e.g. within a warm process).
    if (isServerEnvironment()) {
      // Avoid carrying a worker field on the server (workers are not used)
      const { worker: _ignoredWorker, ...serverSettings } = (settings ||
        {}) as TransformersJSModelSettings & { worker?: unknown };

      const key = getLanguageModelKey(modelId, serverSettings);
      const cached = serverLanguageModelSingletons.get(key);
      if (cached) return cached;

      const instance = new TransformersJSLanguageModel(modelId, serverSettings);
      serverLanguageModelSingletons.set(key, instance);
      return instance;
    }

    return new TransformersJSLanguageModel(modelId, settings);
  };

  const createEmbeddingModel = (
    modelId: TransformersJSEmbeddingModelId,
    settings?: TransformersJSEmbeddingSettings,
  ) => {
    return new TransformersJSEmbeddingModel(modelId, settings);
  };

  const createTranscriptionModel = (
    modelId: TransformersJSTranscriptionModelId,
    settings?: TransformersJSTranscriptionSettings,
  ) => {
    // On the server, return a singleton per model + device + dtype configuration
    // so initialization state persists across uses (e.g. within a warm process).
    if (isServerEnvironment()) {
      const key = getTranscriptionModelKey(modelId, settings);
      const cached = serverTranscriptionModelSingletons.get(key);
      if (cached) return cached;

      const instance = new TransformersJSTranscriptionModel(modelId, settings);
      serverTranscriptionModelSingletons.set(key, instance);
      return instance;
    }

    return new TransformersJSTranscriptionModel(modelId, settings);
  };

  const provider = function (
    modelId: TransformersJSModelId,
    settings?: TransformersJSModelSettings,
  ) {
    if (new.target) {
      throw new Error(
        "The TransformersJS model function cannot be called with the new keyword.",
      );
    }

    return createChatModel(modelId, settings);
  };

  provider.specificationVersion = "v4" as const;
  provider.languageModel = createChatModel;
  provider.chat = createChatModel;
  provider.embedding = createEmbeddingModel;
  provider.embeddingModel = createEmbeddingModel;
  provider.transcription = createTranscriptionModel;
  provider.transcriptionModel = createTranscriptionModel;

  provider.imageModel = (modelId: string) => {
    throw new NoSuchModelError({ modelId, modelType: "imageModel" });
  };

  provider.speechModel = (modelId: string) => {
    throw new NoSuchModelError({ modelId, modelType: "speechModel" });
  };

  return provider;
}

/**
 * Default TransformersJS provider instance.
 */
export const transformersJS = createTransformersJS();

// Server-side singleton cache for language model instances
const serverLanguageModelSingletons = new Map<
  string,
  TransformersJSLanguageModel
>();

// Server-side singleton cache for transcription model instances
const serverTranscriptionModelSingletons = new Map<
  string,
  TransformersJSTranscriptionModel
>();

function getLanguageModelKey(
  modelId: string,
  settings?: TransformersJSModelSettings,
): string {
  const device = (settings?.device ?? "auto").toString();
  const dtype = (settings?.dtype ?? "auto").toString();
  const isVision = !!settings?.isVisionModel;
  return `${modelId}::${device}::${dtype}::${isVision ? "vision" : "text"}`;
}

function getTranscriptionModelKey(
  modelId: string,
  settings?: TransformersJSTranscriptionSettings,
): string {
  const device = (settings?.device ?? "auto").toString();
  const dtype = (settings?.dtype ?? "auto").toString();
  const maxNewTokens = (settings?.maxNewTokens ?? 64).toString();
  return `${modelId}::${device}::${dtype}::${maxNewTokens}`;
}
