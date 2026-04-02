import {
  TextStreamer,
  StoppingCriteriaList,
  StoppingCriteria,
  load_image,
  type PreTrainedTokenizer,
} from "@huggingface/transformers";
import type { TransformersMessage } from "./convert-to-transformers-message";
import type {
  ModelInstance,
  GenerationOptions,
} from "../chat/transformers-js-worker-types";
import type { ToolDefinition, ParsedToolCall } from "@browser-ai/shared";
import { convertToolsToHuggingFaceFormat } from "./convert-tools";

/**
 * Events emitted by the generation stream
 */
export type GenerationEvent =
  | { type: "delta"; delta: string }
  | {
      type: "complete";
      toolCalls?: ParsedToolCall[];
      usage?: { inputTokens?: number; outputTokens?: number };
    };

interface MainThreadOptions {
  modelInstance: ModelInstance;
  messages: TransformersMessage[];
  generationOptions: GenerationOptions;
  tools?: ToolDefinition[];
  isVisionModel?: boolean;
  enableThinking?: boolean;
  stoppingCriteria: StoppingCriteria & {
    interrupt: () => void;
    reset: () => void;
  };
  abortSignal?: AbortSignal;
}

interface WorkerOptions {
  worker: Worker;
  messages: TransformersMessage[];
  generationOptions: GenerationOptions;
  tools?: ToolDefinition[];
  enableThinking?: boolean;
  abortSignal?: AbortSignal;
}

/**
 * Creates an async generator for main thread generation
 */
export async function* createMainThreadGenerationStream(
  options: MainThreadOptions,
): AsyncGenerator<GenerationEvent> {
  const {
    modelInstance,
    messages,
    generationOptions: userGenerationOptions,
    tools,
    isVisionModel,
    enableThinking = false,
    stoppingCriteria,
    abortSignal,
  } = options;

  const [processor, model] = modelInstance;

  const hfTools = tools?.length
    ? convertToolsToHuggingFaceFormat(tools)
    : undefined;

  // Build shared apply_chat_template options
  const templateOptions: Record<string, any> = {
    add_generation_prompt: true,
    ...(hfTools ? { tools: hfTools } : {}),
    ...(enableThinking ? { enable_thinking: true } : {}),
  };

  // Prepare inputs
  let inputs: any;
  let inputLength = 0;

  if (isVisionModel) {
    const text = processor.apply_chat_template(
      messages as any,
      templateOptions,
    );
    const imageUrls = messages
      .flatMap((msg) => (Array.isArray(msg.content) ? msg.content : []))
      .filter((part) => part.type === "image")
      .map((part) => part.image);

    const images = await Promise.all(imageUrls.map((url) => load_image(url)));
    inputs =
      images.length > 0 ? await processor(text, images) : await processor(text);
  } else {
    inputs = processor.apply_chat_template(messages as any, {
      ...templateOptions,
      return_dict: true,
    });
    inputLength = inputs.input_ids.data.length;
  }

  const chunks: Array<GenerationEvent | { type: "error"; error: Error }> = [];
  let resolve: (() => void) | null = null;
  let generationComplete = false;
  let outputTokens = 0;
  let aborted = false;

  const resolvePending = () => {
    resolve?.();
    resolve = null;
  };

  const pushChunk = (
    chunk: GenerationEvent | { type: "error"; error: Error },
  ) => {
    chunks.push(chunk);
    resolvePending();
  };

  const waitForChunk = () =>
    chunks.length > 0 || generationComplete
      ? Promise.resolve()
      : new Promise<void>((r) => {
          resolve = r;
        });

  const abortHandler = () => {
    aborted = true;
    stoppingCriteria.interrupt();
  };

  abortSignal?.addEventListener("abort", abortHandler);

  // Start generation in background
  const generationPromise = (async () => {
    try {
      const streamer = new TextStreamer(
        (isVisionModel
          ? (processor as any).tokenizer
          : processor) as PreTrainedTokenizer,
        {
          skip_prompt: true,
          skip_special_tokens: false,
          callback_function: (text: string) => {
            if (aborted) return;
            // Filter out chat control tokens (e.g. <|im_end|>, <|endoftext|>,
            // <|start_of_turn>, <end_of_turn|>) etc., but preserve tool call and
            // thinking tags.
            const trimmed = text.trim();
            const isSpecialToken =
              /^<\|[^>]+>$/.test(trimmed) || /^<[^|>]+\|>$/.test(trimmed);
            const isToolCallToken =
              trimmed === "<|tool_call|>" ||
              trimmed === "<|tool_call>" ||
              trimmed === "<tool_call|>" ||
              /^<\/?tool_call>$/.test(trimmed);

            if (
              isSpecialToken &&
              !isToolCallToken &&
              !trimmed.includes("channel") // Gemma4 specific
            ) {
              return;
            }

            // Normalize alternative thinking tags to <think></think> so
            // extractReasoningMiddleware({ tagName: "think" }) works for all models.
            if (trimmed === "<|channel>") text = "<think>";
            else if (trimmed === "<channel|>") text = "</think>";
            outputTokens++;
            pushChunk({ type: "delta", delta: text });
          },
        },
      );

      stoppingCriteria.reset();
      const stoppingCriteriaList = new StoppingCriteriaList();
      stoppingCriteriaList.extend([stoppingCriteria]);

      await model.generate({
        ...inputs,
        ...userGenerationOptions,
        streamer,
        stopping_criteria: stoppingCriteriaList,
        return_dict_in_generate: true,
      });

      pushChunk({
        type: "complete",
        usage: { inputTokens: inputLength, outputTokens },
      });
    } catch (error) {
      pushChunk({ type: "error", error: error as Error });
    } finally {
      generationComplete = true;
      resolvePending();
      if (abortSignal) {
        abortSignal.removeEventListener("abort", abortHandler);
      }
    }
  })();

  // Yield chunks as they arrive
  while (true) {
    await waitForChunk();

    while (chunks.length > 0) {
      const chunk = chunks.shift()!;
      if (chunk.type === "error") {
        throw chunk.error;
      }
      yield chunk;
      if (chunk.type === "complete") {
        return;
      }
    }

    if (generationComplete && chunks.length === 0) {
      break;
    }
  }

  await generationPromise;
}

/**
 * Creates an async generator for worker-based generation
 */
export async function* createWorkerGenerationStream(
  options: WorkerOptions,
): AsyncGenerator<GenerationEvent> {
  const {
    worker,
    messages,
    generationOptions,
    tools,
    enableThinking,
    abortSignal,
  } = options;

  const chunks: Array<GenerationEvent | { type: "error"; error: Error }> = [];
  let resolve: (() => void) | null = null;
  let complete = false;

  const pushChunk = (
    chunk: GenerationEvent | { type: "error"; error: Error },
  ) => {
    chunks.push(chunk);
    resolve?.();
    resolve = null;
  };

  const waitForChunk = () =>
    chunks.length > 0 || complete
      ? Promise.resolve()
      : new Promise<void>((r) => {
          resolve = r;
        });

  const onMessage = (e: MessageEvent) => {
    const msg = e.data;
    if (!msg) return;

    if (msg.status === "update" && typeof msg.output === "string") {
      pushChunk({ type: "delta", delta: msg.output });
    } else if (msg.status === "complete") {
      pushChunk({
        type: "complete",
        toolCalls: msg.toolCalls,
        usage: { inputTokens: msg.inputLength, outputTokens: msg.numTokens },
      });
      complete = true;
      worker.removeEventListener("message", onMessage);
    } else if (msg.status === "error") {
      pushChunk({
        type: "error",
        error: new Error(String(msg.data || "Worker error")),
      });
      complete = true;
      worker.removeEventListener("message", onMessage);
    }
  };

  worker.addEventListener("message", onMessage);

  const onAbort = abortSignal
    ? () => {
        worker.postMessage({ type: "interrupt" });
      }
    : null;

  if (abortSignal && onAbort) {
    abortSignal.addEventListener("abort", onAbort);
  }

  worker.postMessage({
    type: "generate",
    data: messages,
    generationOptions,
    tools: tools?.length ? tools : undefined,
    enableThinking,
  });

  // Yield chunks as they arrive
  try {
    while (true) {
      await waitForChunk();

      while (chunks.length > 0) {
        const chunk = chunks.shift()!;
        if (chunk.type === "error") {
          throw chunk.error;
        }
        yield chunk;
        if (chunk.type === "complete") {
          return;
        }
      }

      if (complete && chunks.length === 0) {
        break;
      }
    }
  } finally {
    if (abortSignal && onAbort) {
      abortSignal.removeEventListener("abort", onAbort);
    }
  }
}
