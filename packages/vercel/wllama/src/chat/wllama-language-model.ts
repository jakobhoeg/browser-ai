import {
  LanguageModelV4,
  LanguageModelV4CallOptions,
  LanguageModelV4Content,
  LanguageModelV4FinishReason,
  LanguageModelV4GenerateResult,
  LanguageModelV4StreamPart,
  LanguageModelV4StreamResult,
  LanguageModelV4Usage,
  InvalidPromptError,
  LoadSettingError,
  SharedV4Warning,
} from "@ai-sdk/provider";
import {
  Wllama,
  type ChatCompletionParams,
  type ChatCompletionUsage,
  type LoadModelParams,
} from "@wllama/wllama/esm/index.js";
import {
  createUnsupportedSettingWarning,
  createUnsupportedToolWarning,
  generateToolCallId,
  isFunctionTool,
  type DownloadProgressCallback,
} from "@browser-ai/shared";
import { convertToWllamaMessages } from "../utils/convert-to-wllama-messages";
import {
  convertToWllamaResponseFormat,
  convertToWllamaToolChoice,
  convertToWllamaTools,
} from "../utils/convert-tools";
import { parseModelId } from "../utils/model-source";
import { checkWebGPU, doesBrowserSupportWllama } from "../utils/browser";
import type { Availability } from "../types";

export { doesBrowserSupportWllama };

/**
 * The wllama WASM binary is loaded from jsDelivr. `@wllama/wllama` documents
 * `@wllama/wllama/esm/wasm-from-cdn.js` for this, but that file is missing
 * from the published package. Keep the version in sync with the peer dependency.
 *
 * This is the default (JSPI + Memory64) build. When either is missing (e.g. Safari),
 * wllama switches to its compat build and fetches it from jsDelivr on its own,
 * except on Firefox, where it keeps this build without WebGPU.
 */
const WASM_FROM_CDN = {
  default:
    "https://cdn.jsdelivr.net/npm/@wllama/wllama@3.6.1/src/wasm/wllama.wasm",
};

const DEFAULT_LOAD_PARAMS: LoadModelParams = {
  // Keep <think> blocks inline so `extractReasoningMiddleware` can pick them up
  reasoning_format: "none",
};

/**
 * Model identifier: `hf:owner/repo/path/model.gguf` or `https://.../model.gguf`.
 */
export type WllamaModelId = string;

export interface WllamaSettings {
  /**
   * llama.cpp parameters used when loading the model (n_ctx, n_gpu_layers, ...).
   *
   * Defaults: `{ reasoning_format: "none" }`, plus `n_gpu_layers: 0`
   * (CPU inference) when WebGPU is unavailable. `n_ctx` falls back to
   * wllama's default (1024); raise it for long conversations or tool use.
   */
  loadParams?: LoadModelParams;
  initProgressCallback?: DownloadProgressCallback;
}

type Usage = Pick<ChatCompletionUsage, "prompt_tokens" | "completion_tokens">;

function toUsage(usage?: Usage | null): LanguageModelV4Usage {
  return {
    inputTokens: {
      total: usage?.prompt_tokens,
      noCache: undefined,
      cacheRead: undefined,
      cacheWrite: undefined,
    },
    outputTokens: {
      total: usage?.completion_tokens,
      text: undefined,
      reasoning: undefined,
    },
  };
}

function toFinishReason(
  raw: string | null | undefined,
  hasToolCalls: boolean,
): LanguageModelV4FinishReason {
  if (hasToolCalls) return { unified: "tool-calls", raw: raw ?? undefined };
  switch (raw) {
    case "stop":
      return { unified: "stop", raw };
    case "length":
      return { unified: "length", raw };
    default:
      return { unified: "other", raw: raw ?? undefined };
  }
}

/**
 * llama-server rejects prompts longer than n_ctx with `exceed_context_size_error`,
 * but wllama only forwards its message, e.g. "request (307 tokens) exceeds the
 * available context size (256 tokens), try increasing it".
 */
function toContextSizeError(
  error: unknown,
  prompt: LanguageModelV4CallOptions["prompt"],
): unknown {
  if (
    !(error instanceof Error) ||
    !/exceeds the available context size/.test(error.message)
  ) {
    return error;
  }
  return new InvalidPromptError({
    prompt,
    message: `${error.message}. Shorten the conversation or increase loadParams.n_ctx.`,
    cause: error,
  });
}

function isAbortError(error: unknown): boolean {
  return (
    error instanceof Error &&
    (error.name === "AbortError" || error.name === "WllamaAbortError")
  );
}

export class WllamaLanguageModel implements LanguageModelV4 {
  readonly specificationVersion = "v4";
  readonly provider = "wllama";
  readonly modelId: WllamaModelId;
  readonly supportedUrls: Record<string, RegExp[]> = {};

  private readonly settings: WllamaSettings;
  private engine?: Wllama;
  private initializationPromise?: Promise<Wllama>;
  private unloadingPromise?: Promise<void>;

  constructor(modelId: WllamaModelId, settings: WllamaSettings = {}) {
    this.modelId = modelId;
    this.settings = settings;
  }

  /**
   * "available" once loaded, "downloadable" when the browser supports
   * WebAssembly (GPU or CPU inference), otherwise "unavailable".
   */
  public async availability(): Promise<Availability> {
    if (this.engine) return "available";
    return doesBrowserSupportWllama() ? "downloadable" : "unavailable";
  }

  /**
   * Loads the model, reporting download progress from 0 to 1.
   */
  public async createSessionWithProgress(
    onDownloadProgress?: DownloadProgressCallback,
  ): Promise<WllamaLanguageModel> {
    await this.getEngine(onDownloadProgress);
    return this;
  }

  /**
   * Unloads the model and frees its memory (wasm heap, WebGPU buffers).
   * If the model is loading, waits for it to finish before freeing it;
   * callers waiting for that load receive an AbortError.
   * Call this after generation has stopped.
   * The model is loaded again on the next call.
   */
  public async unload(): Promise<void> {
    if (this.unloadingPromise) return this.unloadingPromise;

    const engine = this.engine;
    const initializationPromise = this.initializationPromise;
    // Invalidate callers waiting for this load so they cannot retain or use
    // the engine after it has been scheduled for disposal.
    this.engine = undefined;
    this.initializationPromise = undefined;
    const unloadingPromise = (async () => {
      // Failed loads are already cleaned up by initializeEngine.
      const loadedEngine =
        engine ?? (await initializationPromise?.catch(() => undefined));
      await loadedEngine?.exit();
    })();
    this.unloadingPromise = unloadingPromise;

    try {
      await unloadingPromise;
    } finally {
      this.unloadingPromise = undefined;
    }
  }

  private async getEngine(
    onDownloadProgress?: DownloadProgressCallback,
  ): Promise<Wllama> {
    // A new request can reload the model once disposal is complete.
    while (this.unloadingPromise) await this.unloadingPromise;
    if (this.engine) return this.engine;

    if (!doesBrowserSupportWllama()) {
      throw new LoadSettingError({
        message: "wllama Engine initialization failed.",
      });
    }

    // Concurrent callers share one load; a failed load can be retried.
    const initializationPromise = (this.initializationPromise ??=
      this.initializeEngine(
        onDownloadProgress ?? this.settings.initProgressCallback,
      ).catch((error) => {
        if (this.initializationPromise === initializationPromise) {
          this.initializationPromise = undefined;
        }
        throw new LoadSettingError({
          message: `Failed to initialize wllama engine: ${error instanceof Error ? error.message : "Unknown error"}`,
        });
      }));

    const engine = await initializationPromise;
    if (this.initializationPromise !== initializationPromise) {
      throw new DOMException("Model unloaded while loading", "AbortError");
    }
    this.engine = engine;
    return engine;
  }

  private async initializeEngine(
    onProgress?: DownloadProgressCallback,
  ): Promise<Wllama> {
    const source = parseModelId(this.modelId);
    const params = {
      ...DEFAULT_LOAD_PARAMS,
      ...(checkWebGPU() ? {} : { n_gpu_layers: 0 }),
      ...this.settings.loadParams,
      progressCallback: ({
        loaded,
        total,
      }: {
        loaded: number;
        total: number;
      }) => onProgress?.(total > 0 ? Math.min(loaded / total, 1) : 0),
    };

    const engine = new Wllama(WASM_FROM_CDN);
    try {
      if (source.type === "url") {
        await engine.loadModelFromUrl(source.url, params);
      } else {
        await engine.loadModelFromHF(source.params, params);
      }
      return engine;
    } catch (error) {
      // Loading can fail after starting the worker or allocating GPU memory.
      // Preserve the load error even if disposal also fails.
      await engine.exit().catch(() => undefined);
      throw error;
    }
  }

  private getArgs(options: LanguageModelV4CallOptions) {
    const warnings: SharedV4Warning[] = [];

    for (const tool of options.tools ?? []) {
      if (!isFunctionTool(tool)) {
        warnings.push(
          createUnsupportedToolWarning(
            tool,
            "Only function tools are supported by wllama",
          ),
        );
      }
    }
    const functionTools = (options.tools ?? []).filter(isFunctionTool);

    const unsupported: Array<[string, unknown]> = [
      ["stopSequences", options.stopSequences?.length || undefined],
      ["frequencyPenalty", options.frequencyPenalty],
      ["presencePenalty", options.presencePenalty],
    ];
    for (const [name, value] of unsupported) {
      if (value != null) {
        warnings.push(
          createUnsupportedSettingWarning(
            name,
            `${name} is not supported by wllama and is ignored`,
          ),
        );
      }
    }

    const request: ChatCompletionParams = {
      messages: convertToWllamaMessages(options.prompt),
      max_tokens: options.maxOutputTokens,
      temperature: options.temperature,
      top_p: options.topP,
      top_k: options.topK,
      seed: options.seed,
      response_format: convertToWllamaResponseFormat(options.responseFormat),
    };

    if (functionTools.length > 0) {
      request.tools = convertToWllamaTools(functionTools);
      request.tool_choice = convertToWllamaToolChoice(options.toolChoice);
    }

    return { request, warnings };
  }

  public async doGenerate(
    options: LanguageModelV4CallOptions,
  ): Promise<LanguageModelV4GenerateResult> {
    const { request, warnings } = this.getArgs(options);
    const engine = await this.getEngine();

    const response = await engine
      .createChatCompletion({
        ...request,
        stream: false,
        abortSignal: options.abortSignal,
      })
      .catch((error) => {
        throw toContextSizeError(error, options.prompt);
      });

    const choice = response.choices[0];
    if (!choice) {
      throw new Error("No response choice returned from wllama");
    }

    const content: LanguageModelV4Content[] = [];
    if (choice.message.content) {
      content.push({ type: "text", text: choice.message.content });
    }
    const toolCalls = choice.message.tool_calls ?? [];
    for (const call of toolCalls) {
      content.push({
        type: "tool-call",
        toolCallId: call.id || generateToolCallId(),
        toolName: call.function.name,
        input: call.function.arguments || "{}",
      });
    }

    return {
      content,
      finishReason: toFinishReason(choice.finish_reason, toolCalls.length > 0),
      usage: toUsage(response.usage),
      request: { body: request },
      warnings,
    };
  }

  public async doStream(
    options: LanguageModelV4CallOptions,
  ): Promise<LanguageModelV4StreamResult> {
    const { request, warnings } = this.getArgs(options);
    const engine = await this.getEngine();

    const stream = new ReadableStream<LanguageModelV4StreamPart>({
      async start(controller) {
        const textId = "text-0";
        let textStarted = false;
        // Tool calls arrive as OpenAI-style deltas keyed by index; they are
        // accumulated and emitted once the model finishes.
        const toolCalls = new Map<
          number,
          { id: string; name: string; args: string }
        >();
        let usage: Usage | null | undefined;
        let rawFinishReason: string | null | undefined;

        controller.enqueue({ type: "stream-start", warnings });

        try {
          const chunks = await engine.createChatCompletion({
            ...request,
            stream: true,
            abortSignal: options.abortSignal,
          });

          for await (const chunk of chunks) {
            if (chunk.usage) usage = chunk.usage;
            const choice = chunk.choices[0];
            if (!choice) continue;

            if (choice.delta.content) {
              if (!textStarted) {
                controller.enqueue({ type: "text-start", id: textId });
                textStarted = true;
              }
              controller.enqueue({
                type: "text-delta",
                id: textId,
                delta: choice.delta.content,
              });
            }

            for (const delta of choice.delta.tool_calls ?? []) {
              const call = toolCalls.get(delta.index) ?? {
                id: "",
                name: "",
                args: "",
              };
              call.id ||= delta.id ?? "";
              call.name ||= delta.function?.name ?? "";
              call.args += delta.function?.arguments ?? "";
              toolCalls.set(delta.index, call);
            }

            if (choice.finish_reason) rawFinishReason = choice.finish_reason;
          }
        } catch (error) {
          if (!isAbortError(error)) {
            controller.error(toContextSizeError(error, options.prompt));
            return;
          }
          rawFinishReason = "abort";
        }

        if (textStarted) {
          controller.enqueue({ type: "text-end", id: textId });
        }

        let hasToolCalls = false;
        for (const call of toolCalls.values()) {
          if (!call.name) continue;
          hasToolCalls = true;
          const id = call.id || generateToolCallId();
          const input = call.args || "{}";
          controller.enqueue({
            type: "tool-input-start",
            id,
            toolName: call.name,
          });
          controller.enqueue({ type: "tool-input-delta", id, delta: input });
          controller.enqueue({ type: "tool-input-end", id });
          controller.enqueue({
            type: "tool-call",
            toolCallId: id,
            toolName: call.name,
            input,
          });
        }

        controller.enqueue({
          type: "finish",
          finishReason: toFinishReason(rawFinishReason, hasToolCalls),
          usage: toUsage(usage),
        });
        controller.close();
      },
    });

    return { stream, request: { body: request } };
  }
}
