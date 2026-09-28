import {
  LanguageModelV4,
  LanguageModelV4CallOptions,
  LanguageModelV4Content,
  LanguageModelV4FinishReason,
  LanguageModelV4GenerateResult,
  LanguageModelV4StreamPart,
  LanguageModelV4StreamResult,
  LoadSettingError,
  SharedV4Warning,
} from "@ai-sdk/provider";
import {
  createUnsupportedSettingWarning,
  isAutoToolChoice,
  type DownloadProgressCallback,
} from "@browser-ai/shared";
import { Engine, type EngineSettings } from "@litert-lm/core";

import { type Availability } from "../types";
import { convertToLiteRTMessages } from "../utils/convert-to-litert-messages";
import { checkWebGPU, doesBrowserSupportLiteRTLM } from "../utils/browser";

export { doesBrowserSupportLiteRTLM };

/**
 * A model id for LiteRT-LM is the location of a `.litertlm` model artifact:
 * an HTTPS URL, a local path, a `Blob`, or a `ReadableStream`. Convenience
 * constants for the models supported by the Web SDK are exported from the
 * package root (`GEMMA_4_E2B_WEB`, `GEMMA_4_E4B_WEB`).
 */
export type LiteRTLMModelId = string;

export interface LiteRTLMSettings {
  /**
   * Raw `EngineSettings` overrides forwarded to `Engine.create`. Use this to
   * tune `mainExecutorSettings` (e.g. `maxNumTokens`) or pass a custom model
   * `Blob`/`ReadableStream`.
   */
  engineSettings?: Partial<EngineSettings>;
  /**
   * Enable reasoning ("thinking") tokens for reasoning-capable models. Maps to
   * `extra_context.enable_thinking` in the conversation preface.
   *
   * @default false
   */
  enableThinking?: boolean;
}

type LiteRTEngine = Awaited<ReturnType<typeof Engine.create>>;
type LiteRTConversation = Awaited<
  ReturnType<LiteRTEngine["createConversation"]>
>;

type LiteRTConfig = {
  provider: string;
  modelId: LiteRTLMModelId;
  options: LiteRTLMSettings;
};

const DEFAULT_MAX_NUM_TOKENS = 8192;

export class LiteRTLMLanguageModel implements LanguageModelV4 {
  readonly specificationVersion = "v4";
  readonly modelId: LiteRTLMModelId;
  readonly provider = "litert-lm";

  private readonly config: LiteRTConfig;
  private engine?: LiteRTEngine;
  private isInitialized = false;
  private initializationPromise?: Promise<void>;

  constructor(modelId: LiteRTLMModelId, options: LiteRTLMSettings = {}) {
    this.modelId = modelId;
    this.config = {
      provider: this.provider,
      modelId,
      options,
    };
  }

  readonly supportedUrls: Record<string, RegExp[]> = {
    // LiteRT-LM resolves models from explicit URLs/paths, not provider URLs.
  };

  get isModelInitialized(): boolean {
    return this.isInitialized;
  }

  private async getEngine(): Promise<LiteRTEngine> {
    const availability = await this.availability();
    if (availability === "unavailable") {
      throw new LoadSettingError({
        message:
          "LiteRT-LM is not available. This provider requires a browser with WebGPU support.",
      });
    }

    if (this.engine && this.isInitialized) return this.engine;

    if (this.initializationPromise) {
      await this.initializationPromise;
      if (this.engine) return this.engine;
    }

    this.initializationPromise = this._initializeEngine();
    await this.initializationPromise;

    if (!this.engine) {
      throw new LoadSettingError({ message: "Engine initialization failed" });
    }

    return this.engine;
  }

  private async _initializeEngine(): Promise<void> {
    const base: Record<string, unknown> = {
      model: this.modelId,
      ...this.config.options.engineSettings,
    };

    if (!base.mainExecutorSettings) {
      base.mainExecutorSettings = { maxNumTokens: DEFAULT_MAX_NUM_TOKENS };
    }

    try {
      this.engine = await Engine.create(base as EngineSettings);
      this.isInitialized = true;
    } catch (error) {
      this.engine = undefined;
      this.isInitialized = false;
      this.initializationPromise = undefined;

      throw new LoadSettingError({
        message: `Failed to initialize LiteRT-LM engine: ${
          error instanceof Error ? error.message : "Unknown error"
        }`,
      });
    }
  }

  private getArgs(options: LanguageModelV4CallOptions) {
    const warnings: SharedV4Warning[] = [];

    const tools = options.tools ?? [];

    if (tools.length > 0) {
      warnings.push(
        createUnsupportedSettingWarning(
          "tools",
          "Tool/function calling is not yet supported by the LiteRT-LM Web SDK",
        ),
      );
    }

    if (options.toolChoice != null && !isAutoToolChoice(options.toolChoice)) {
      warnings.push(
        createUnsupportedSettingWarning(
          "toolChoice",
          "toolChoice is not supported by LiteRT-LM",
        ),
      );
    }

    if (options.stopSequences != null) {
      warnings.push(
        createUnsupportedSettingWarning(
          "stopSequences",
          "stopSequences is not supported by LiteRT-LM",
        ),
      );
    }

    if (options.frequencyPenalty != null) {
      warnings.push(
        createUnsupportedSettingWarning(
          "frequencyPenalty",
          "frequencyPenalty is not supported by LiteRT-LM",
        ),
      );
    }

    if (options.presencePenalty != null) {
      warnings.push(
        createUnsupportedSettingWarning(
          "presencePenalty",
          "presencePenalty is not supported by LiteRT-LM",
        ),
      );
    }

    if (options.topK != null) {
      warnings.push(
        createUnsupportedSettingWarning(
          "topK",
          "topK is not supported by LiteRT-LM",
        ),
      );
    }

    if (options.seed != null) {
      warnings.push(
        createUnsupportedSettingWarning(
          "seed",
          "seed is not supported by LiteRT-LM",
        ),
      );
    }

    if (options.responseFormat?.type === "json") {
      warnings.push(
        createUnsupportedSettingWarning(
          "responseFormat",
          "JSON response format is not yet supported by the LiteRT-LM Web SDK",
        ),
      );
    }

    const allMessages = convertToLiteRTMessages(options.prompt);
    // Everything except the final message becomes conversation history; only
    // the last turn is actually generated.
    const history = allMessages.slice(0, -1);
    const lastMessage =
      allMessages[allMessages.length - 1] ?? { role: "user" as const, content: "" };

    const providerOptions = options.providerOptions?.[this.provider];
    const enableThinking =
      this.config.options.enableThinking ??
      (providerOptions?.enableThinking as boolean | undefined) ??
      false;

    return {
      warnings,
      allMessages,
      history,
      lastMessage,
      enableThinking,
      maxOutputTokens: options.maxOutputTokens,
    };
  }

  public async doGenerate(
    options: LanguageModelV4CallOptions,
  ): Promise<LanguageModelV4GenerateResult> {
    const { warnings, allMessages, history, lastMessage, enableThinking, maxOutputTokens } =
      this.getArgs(options);

    const engine = await this.getEngine();

    const preface: Record<string, unknown> = { messages: history };
    if (enableThinking) preface.extra_context = { enable_thinking: true };

    const conversation = (await engine.createConversation({
      preface,
      ...(maxOutputTokens != null
        ? { sessionConfig: { maxOutputTokens } }
        : {}),
    } as never)) as LiteRTConversation;

    const abortHandler = () => {
      conversation.cancel();
    };
    if (options.abortSignal) {
      options.abortSignal.addEventListener("abort", abortHandler);
    }

    try {
      const response = await conversation.sendMessage({
        role: "user",
        content: lastMessage.content,
      } as never);

      const text =
        (response as { content?: Array<{ text?: string }> })?.content?.[0]
          ?.text ?? "";

      const content: LanguageModelV4Content[] = [
        {
          type: "text",
          text,
        },
      ];

      return {
        content,
        finishReason: { unified: "stop", raw: "stop" },
        usage: {
          inputTokens: {
            total: undefined,
            noCache: undefined,
            cacheRead: undefined,
            cacheWrite: undefined,
          },
          outputTokens: {
            total: undefined,
            text: undefined,
            reasoning: undefined,
          },
        },
        request: { body: { messages: allMessages, lastMessage } },
        warnings,
      };
    } catch (error) {
      throw new Error(
        `LiteRT-LM generation failed: ${
          error instanceof Error ? error.message : "Unknown error"
        }`,
      );
    } finally {
      if (options.abortSignal) {
        options.abortSignal.removeEventListener("abort", abortHandler);
      }
    }
  }

  public async availability(): Promise<Availability> {
    if (this.isInitialized && this.engine) return "available";
    return checkWebGPU() ? "downloadable" : "unavailable";
  }

  /**
   * Creates an engine session, optionally reporting download progress.
   *
   * Note: the LiteRT-LM Web SDK does not currently expose a streaming model
   * download progress callback, so we only report start (0) and completion (1).
   */
  public async createSessionWithProgress(
    onDownloadProgress?: DownloadProgressCallback,
  ): Promise<LiteRTLMLanguageModel> {
    onDownloadProgress?.(0);
    await this.getEngine();
    onDownloadProgress?.(1);
    return this;
  }

  public async doStream(
    options: LanguageModelV4CallOptions,
  ): Promise<LanguageModelV4StreamResult> {
    const { warnings, allMessages, history, lastMessage, enableThinking, maxOutputTokens } =
      this.getArgs(options);

    const engine = await this.getEngine();

    const preface: Record<string, unknown> = { messages: history };
    if (enableThinking) preface.extra_context = { enable_thinking: true };

    const conversation = (await engine.createConversation({
      preface,
      ...(maxOutputTokens != null
        ? { sessionConfig: { maxOutputTokens } }
        : {}),
    } as never)) as LiteRTConversation;

    const abortHandler = () => {
      conversation.cancel();
    };
    if (options.abortSignal) {
      options.abortSignal.addEventListener("abort", abortHandler);
    }

    const textId = "text-0";

    const stream = new ReadableStream<LanguageModelV4StreamPart>({
      async start(controller) {
        controller.enqueue({
          type: "stream-start",
          warnings,
        });

        let textStarted = false;
        let finished = false;
        let isAbort = false;

        const ensureTextStart = () => {
          if (!textStarted) {
            controller.enqueue({ type: "text-start", id: textId });
            textStarted = true;
          }
        };

        const emitTextDelta = (delta: string) => {
          if (!delta) return;
          ensureTextStart();
          controller.enqueue({
            type: "text-delta",
            id: textId,
            delta,
          });
        };

        const emitTextEndIfNeeded = () => {
          if (!textStarted) return;
          controller.enqueue({ type: "text-end", id: textId });
          textStarted = false;
        };

        const finishStream = (finishReason: LanguageModelV4FinishReason) => {
          if (finished) return;
          finished = true;
          emitTextEndIfNeeded();
          controller.enqueue({
            type: "finish",
            finishReason,
            usage: {
              inputTokens: {
                total: undefined,
                noCache: undefined,
                cacheRead: undefined,
                cacheWrite: undefined,
              },
              outputTokens: {
                total: undefined,
                text: undefined,
                reasoning: undefined,
              },
            },
          });
          controller.close();
        };

        try {
          const chunks = conversation.sendMessageStreaming(
            lastMessage.content,
          );

          for await (const chunk of chunks) {
            if (options.abortSignal?.aborted) {
              isAbort = true;
              break;
            }

            const content = (chunk as { content?: unknown }).content;
            if (Array.isArray(content)) {
              for (const item of content) {
                const maybe = item as { type?: string; text?: string };
                if (maybe?.type === "text" && typeof maybe.text === "string") {
                  emitTextDelta(maybe.text);
                }
              }
            } else if (typeof (chunk as { text?: unknown }).text === "string") {
              emitTextDelta((chunk as { text: string }).text);
            }
          }

          finishStream(
            isAbort
              ? { unified: "other", raw: "abort" }
              : { unified: "stop", raw: "stop" },
          );
        } catch (error) {
          if (!finished) {
            finished = true;
            controller.error(error);
          }
        } finally {
          if (options.abortSignal) {
            options.abortSignal.removeEventListener("abort", abortHandler);
          }
        }
      },
    });

    return {
      stream,
      request: { body: { messages: allMessages, lastMessage } },
    };
  }
}
