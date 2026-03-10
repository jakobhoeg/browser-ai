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
  chatTemplateOptions?: Record<string, unknown>;
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
    chatTemplateOptions,
    stoppingCriteria,
    abortSignal,
  } = options;

  const [processor, model] = modelInstance;

  const hfTools = !isVisionModel && tools?.length
    ? convertToolsToHuggingFaceFormat(tools)
    : undefined;

  // Prepare inputs
  let inputs: any;
  let inputLength = 0;

  if (isVisionModel) {
    const text = processor.apply_chat_template(messages as any, {
      add_generation_prompt: true,
      ...chatTemplateOptions,
    });
    const imageUrls = messages
      .flatMap((msg) => (Array.isArray(msg.content) ? msg.content : []))
      .filter((part) => part.type === "image")
      .map((part) => part.image);

    const images = await Promise.all(imageUrls.map(async (url) => {
      const img = await load_image(url);
      return img.resize(448, 448);
    }));
    inputs = images.length > 0 ? await processor(text, images) : await processor(text);
  } else {
    inputs = processor.apply_chat_template(messages as any, {
      add_generation_prompt: true,
      return_dict: true,
      ...(hfTools ? { tools: hfTools } : {}),
      ...chatTemplateOptions,
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

  const enableThinking = !!chatTemplateOptions?.enable_thinking;

  // Emit opening <think> tag since the template injects it as a prompt prefix,
  // not as generated output, so the streamer never sees it.
  if (enableThinking) {
    pushChunk({ type: "delta", delta: "<think>" });
  }

  // Start generation in background
  const generationPromise = (async () => {
    try {
      const streamer = new TextStreamer(
        (isVisionModel
          ? (processor as any).tokenizer
          : processor) as PreTrainedTokenizer,
        {
          skip_prompt: true,
          skip_special_tokens: !enableThinking,
          callback_function: (text: string) => {
            if (aborted) return;
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
  const { worker, messages, generationOptions, tools, abortSignal } = options;

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
