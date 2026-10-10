import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { generateText, streamText, generateObject } from "ai";
import { InvalidPromptError, LoadSettingError } from "@ai-sdk/provider";
import { z } from "zod";
import { WllamaLanguageModel, wllama } from "../src";
import packageJson from "../package.json";

const MODEL_ID = "hf:owner/repo/model.gguf";
// Message wllama forwards from llama-server's exceed_context_size_error
const CONTEXT_SIZE_MESSAGE =
  "request (307 tokens) exceeds the available context size (256 tokens), try increasing it";

const mockCreateChatCompletion = vi.fn();
const mockLoadModelFromHF = vi.fn();
const mockLoadModelFromUrl = vi.fn();
const mockExit = vi.fn();
const mockConstructor = vi.fn();

vi.mock("@wllama/wllama/esm/index.js", () => ({
  Wllama: vi.fn().mockImplementation(function (
    pathConfig: unknown,
    config: unknown,
  ) {
    mockConstructor(pathConfig, config);
    return {
      loadModelFromHF: mockLoadModelFromHF,
      loadModelFromUrl: mockLoadModelFromUrl,
      createChatCompletion: mockCreateChatCompletion,
      exit: mockExit,
    };
  }),
}));

function chatResponse(
  overrides: Partial<{
    content: string | null;
    tool_calls: Array<{
      id: string;
      type: "function";
      function: { name: string; arguments: string };
    }>;
    finish_reason: "stop" | "length" | "tool_calls" | "content_filter" | null;
  }> = {},
) {
  const { content = "Hello!", tool_calls, finish_reason = "stop" } = overrides;
  return {
    id: "chatcmpl-1",
    object: "chat.completion",
    created: 0,
    model: "test",
    choices: [
      {
        index: 0,
        message: { role: "assistant", content, tool_calls },
        finish_reason,
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
  };
}

async function* chunks(
  parts: Array<{
    content?: string;
    tool_calls?: Array<{
      index: number;
      id?: string;
      function?: { name?: string; arguments?: string };
    }>;
    finish_reason?: "stop" | "length" | "tool_calls" | null;
    usage?: {
      prompt_tokens: number;
      completion_tokens: number;
      total_tokens: number;
    };
  }>,
) {
  for (const part of parts) {
    yield {
      id: "chatcmpl-1",
      object: "chat.completion.chunk",
      created: 0,
      model: "test",
      choices: [
        {
          index: 0,
          delta: {
            content: part.content,
            tool_calls: part.tool_calls,
          },
          finish_reason: part.finish_reason ?? null,
          logprobs: null,
        },
      ],
      usage: part.usage,
    };
  }
}

const userPrompt = (text: string) => [
  { role: "user" as const, content: [{ type: "text" as const, text }] },
];

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

describe("WllamaLanguageModel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockLoadModelFromHF.mockResolvedValue(undefined);
    mockLoadModelFromUrl.mockResolvedValue(undefined);
    mockExit.mockReset().mockResolvedValue(undefined);
    Object.defineProperty(global.navigator, "gpu", {
      value: {},
      configurable: true,
    });
    (globalThis.WebAssembly as any).Suspending = function Suspending() {};
  });

  afterEach(() => {
    Object.defineProperty(global.navigator, "gpu", {
      value: undefined,
      configurable: true,
    });
  });

  describe("constructor", () => {
    it("creates a model instance", () => {
      const model = new WllamaLanguageModel("hf:owner/repo/model.gguf");
      expect(model).toBeInstanceOf(WllamaLanguageModel);
      expect(model.specificationVersion).toBe("v4");
      expect(model.provider).toBe("wllama");
      expect(model.modelId).toBe("hf:owner/repo/model.gguf");
    });

    it("creates a model using the factory function", () => {
      const model = wllama("hf:owner/repo/model.gguf");
      expect(model).toBeInstanceOf(WllamaLanguageModel);
    });

    it("cannot be called with new", () => {
      expect(() => new (wllama as any)("x/y")).toThrow();
    });
  });

  describe("availability", () => {
    it("returns 'downloadable' without WebGPU (CPU inference)", async () => {
      Object.defineProperty(global.navigator, "gpu", {
        value: undefined,
        configurable: true,
      });
      const model = new WllamaLanguageModel(MODEL_ID);
      expect(await model.availability()).toBe("downloadable");
    });

    it("returns 'downloadable' before loading and 'available' after", async () => {
      const model = new WllamaLanguageModel(MODEL_ID);
      expect(await model.availability()).toBe("downloadable");
      await model.createSessionWithProgress();
      expect(await model.availability()).toBe("available");
    });
  });

  describe("model loading", () => {
    it("loads Hugging Face repo/file ids with default params", async () => {
      const model = new WllamaLanguageModel(
        "hf:ggml-org/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf",
      );
      await model.createSessionWithProgress();

      const wllamaVersion = packageJson.peerDependencies["@wllama/wllama"];
      expect(wllamaVersion).toMatch(/^\d+\.\d+\.\d+$/);
      expect(mockConstructor).toHaveBeenCalledWith(
        {
          default: `https://cdn.jsdelivr.net/npm/@wllama/wllama@${wllamaVersion}/src/wasm/wllama.wasm`,
        },
        undefined,
      );
      expect(mockLoadModelFromHF).toHaveBeenCalledTimes(1);
      const [hfParams, loadParams] = mockLoadModelFromHF.mock.calls[0];
      expect(hfParams).toEqual({
        repo: "ggml-org/Qwen3-0.6B-GGUF",
        file: "Qwen3-0.6B-Q8_0.gguf",
      });
      expect(loadParams).toMatchObject({ reasoning_format: "none" });
      expect(loadParams.n_ctx).toBeUndefined();
      expect(loadParams.n_gpu_layers).toBeUndefined();
      expect(typeof loadParams.progressCallback).toBe("function");
    });

    it("runs on the CPU when WebGPU is unavailable", async () => {
      Object.defineProperty(global.navigator, "gpu", {
        value: undefined,
        configurable: true,
      });
      const model = new WllamaLanguageModel(MODEL_ID);
      await model.createSessionWithProgress();
      expect(mockLoadModelFromHF.mock.calls[0][1].n_gpu_layers).toBe(0);
    });

    it("keeps GPU inference without JSPI (Asyncify)", async () => {
      const original = (globalThis.WebAssembly as any).Suspending;
      delete (globalThis.WebAssembly as any).Suspending;
      try {
        const model = new WllamaLanguageModel(MODEL_ID);
        await model.createSessionWithProgress();
        expect(
          mockLoadModelFromHF.mock.calls[0][1].n_gpu_layers,
        ).toBeUndefined();
      } finally {
        (globalThis.WebAssembly as any).Suspending = original;
      }
    });

    it("lets loadParams override the CPU fallback", async () => {
      Object.defineProperty(global.navigator, "gpu", {
        value: undefined,
        configurable: true,
      });
      const model = new WllamaLanguageModel(MODEL_ID, {
        loadParams: { n_gpu_layers: 10 },
      });
      await model.createSessionWithProgress();
      expect(mockLoadModelFromHF.mock.calls[0][1].n_gpu_layers).toBe(10);
    });

    it("loads direct URLs and forwards loadParams", async () => {
      const model = new WllamaLanguageModel("https://example.com/m.gguf", {
        loadParams: { n_ctx: 2048, n_gpu_layers: 10 },
      });
      await model.createSessionWithProgress();

      expect(mockLoadModelFromUrl).toHaveBeenCalledWith(
        "https://example.com/m.gguf",
        expect.objectContaining({ n_ctx: 2048, n_gpu_layers: 10 }),
      );
    });

    it("normalizes download progress to 0..1", async () => {
      const progress: number[] = [];
      mockLoadModelFromHF.mockImplementation(async (_hf, params) => {
        params.progressCallback({ loaded: 50, total: 200 });
        params.progressCallback({ loaded: 200, total: 200 });
      });
      const model = new WllamaLanguageModel(MODEL_ID);
      await model.createSessionWithProgress((p) => progress.push(p));
      expect(progress).toEqual([0.25, 1]);
    });

    it("shares one initialization across concurrent callers", async () => {
      mockCreateChatCompletion.mockResolvedValue(chatResponse());
      const model = new WllamaLanguageModel(MODEL_ID);
      await Promise.all([
        model.doGenerate({ prompt: userPrompt("a") }),
        model.doGenerate({ prompt: userPrompt("b") }),
      ]);
      expect(mockConstructor).toHaveBeenCalledTimes(1);
      expect(mockLoadModelFromHF).toHaveBeenCalledTimes(1);
    });

    it("wraps load failures in LoadSettingError and allows retry", async () => {
      mockLoadModelFromHF.mockRejectedValueOnce(new Error("boom"));
      const model = new WllamaLanguageModel(MODEL_ID);
      await expect(model.createSessionWithProgress()).rejects.toBeInstanceOf(
        LoadSettingError,
      );
      expect(mockExit).toHaveBeenCalledTimes(1);
      await model.createSessionWithProgress();
      expect(mockLoadModelFromHF).toHaveBeenCalledTimes(2);
    });

    it("cleans up a failed URL load and preserves its error if exit fails", async () => {
      mockLoadModelFromUrl.mockRejectedValueOnce(
        new Error("GPU allocation failed"),
      );
      mockExit.mockRejectedValueOnce(new Error("exit failed"));
      const model = new WllamaLanguageModel("https://example.com/model.gguf");

      await expect(model.createSessionWithProgress()).rejects.toThrow(
        "GPU allocation failed",
      );
      await model.unload();
      expect(mockExit).toHaveBeenCalledTimes(1);
      await model.createSessionWithProgress();
      expect(await model.availability()).toBe("available");
    });
  });

  describe("unload", () => {
    it("exits the engine and reloads on the next call", async () => {
      const model = new WllamaLanguageModel(MODEL_ID);
      await model.createSessionWithProgress();

      await model.unload();
      expect(mockExit).toHaveBeenCalledTimes(1);
      expect(await model.availability()).toBe("downloadable");

      await model.createSessionWithProgress();
      expect(mockConstructor).toHaveBeenCalledTimes(2);
      expect(mockLoadModelFromHF).toHaveBeenCalledTimes(2);
    });

    it("does nothing when the model is not loaded", async () => {
      const model = new WllamaLanguageModel(MODEL_ID);
      await model.unload();
      expect(mockExit).not.toHaveBeenCalled();
    });

    it("waits for loading, releases the engine once, and aborts pending generation", async () => {
      const load = deferred<void>();
      mockLoadModelFromHF.mockReturnValueOnce(load.promise);
      const model = new WllamaLanguageModel(MODEL_ID);
      const generation = model.doGenerate({ prompt: userPrompt("hi") });
      const aborted = expect(generation).rejects.toMatchObject({
        name: "AbortError",
      });
      const unloading = model.unload();
      const concurrentUnloading = model.unload();

      expect(mockExit).not.toHaveBeenCalled();
      load.resolve(undefined);
      await Promise.all([aborted, unloading, concurrentUnloading]);

      expect(mockExit).toHaveBeenCalledTimes(1);
      expect(mockCreateChatCompletion).not.toHaveBeenCalled();
      expect(await model.availability()).toBe("downloadable");

      await model.createSessionWithProgress();
      expect(mockConstructor).toHaveBeenCalledTimes(2);
      expect(await model.availability()).toBe("available");
    });

    it("delays a new load until the old engine has exited", async () => {
      const load = deferred<void>();
      const exit = deferred<void>();
      mockLoadModelFromHF.mockReturnValueOnce(load.promise);
      mockExit.mockReturnValueOnce(exit.promise);
      const model = new WllamaLanguageModel(MODEL_ID);
      const loading = model.createSessionWithProgress();
      const aborted = expect(loading).rejects.toMatchObject({
        name: "AbortError",
      });
      const unloading = model.unload();
      const reloading = model.createSessionWithProgress();

      load.resolve(undefined);
      await aborted;
      expect(mockExit).toHaveBeenCalledTimes(1);
      expect(mockLoadModelFromHF).toHaveBeenCalledTimes(1);
      expect(await model.availability()).toBe("downloadable");

      exit.resolve(undefined);
      await Promise.all([unloading, reloading]);
      expect(mockLoadModelFromHF).toHaveBeenCalledTimes(2);
      expect(await model.availability()).toBe("available");
    });

    it("cleans up a failed pending load once and allows the next load", async () => {
      const load = deferred<void>();
      mockLoadModelFromHF.mockReturnValueOnce(load.promise);
      const model = new WllamaLanguageModel(MODEL_ID);
      const loading = model.createSessionWithProgress();
      const failed = expect(loading).rejects.toThrow("load failed");
      const unloading = model.unload();
      const reloading = model.createSessionWithProgress();

      load.reject(new Error("load failed"));
      await Promise.all([failed, unloading, reloading]);

      expect(mockExit).toHaveBeenCalledTimes(1);
      expect(mockLoadModelFromHF).toHaveBeenCalledTimes(2);
      expect(await model.availability()).toBe("available");
    });
  });

  describe("doGenerate", () => {
    it("generates text via generateText", async () => {
      mockCreateChatCompletion.mockResolvedValue(
        chatResponse({ content: "Hello world" }),
      );
      const model = new WllamaLanguageModel(MODEL_ID);

      const result = await generateText({
        model,
        prompt: "Say hi",
        temperature: 0.3,
        maxOutputTokens: 42,
        topK: 20,
      });

      expect(result.text).toBe("Hello world");
      expect(result.usage.inputTokens).toBe(10);
      expect(result.usage.outputTokens).toBe(5);
      expect(result.finishReason).toBe("stop");

      const request = mockCreateChatCompletion.mock.calls[0][0];
      expect(request).toMatchObject({
        stream: false,
        temperature: 0.3,
        max_tokens: 42,
        top_k: 20,
        messages: [{ role: "user", content: "Say hi" }],
      });
      expect(request.tools).toBeUndefined();
    });

    it("maps native tool calls", async () => {
      mockCreateChatCompletion.mockResolvedValue(
        chatResponse({
          content: null,
          finish_reason: "tool_calls",
          tool_calls: [
            {
              id: "call_1",
              type: "function",
              function: {
                name: "getWeather",
                arguments: '{"city":"Tokyo"}',
              },
            },
          ],
        }),
      );
      const model = new WllamaLanguageModel(MODEL_ID);

      const result = await model.doGenerate({
        prompt: userPrompt("Weather in Tokyo?"),
        tools: [
          {
            type: "function",
            name: "getWeather",
            description: "Get weather",
            inputSchema: {
              type: "object",
              properties: { city: { type: "string" } },
              required: ["city"],
            },
          },
        ],
        toolChoice: { type: "required" },
      });

      expect(result.finishReason).toEqual({
        unified: "tool-calls",
        raw: "tool_calls",
      });
      expect(result.content).toEqual([
        {
          type: "tool-call",
          toolCallId: "call_1",
          toolName: "getWeather",
          input: '{"city":"Tokyo"}',
        },
      ]);

      const request = mockCreateChatCompletion.mock.calls[0][0];
      expect(request.tools).toEqual([
        {
          type: "function",
          function: {
            name: "getWeather",
            description: "Get weather",
            parameters: {
              type: "object",
              properties: { city: { type: "string" } },
              required: ["city"],
            },
          },
        },
      ]);
      expect(request.tool_choice).toBe("required");
    });

    it("sends assistant tool calls and tool results in native format", async () => {
      mockCreateChatCompletion.mockResolvedValue(
        chatResponse({ content: "It is sunny." }),
      );
      const model = new WllamaLanguageModel(MODEL_ID);

      await model.doGenerate({
        prompt: [
          ...userPrompt("Weather?"),
          {
            role: "assistant",
            content: [
              {
                type: "tool-call",
                toolCallId: "call_1",
                toolName: "getWeather",
                input: { city: "Tokyo" },
              },
            ],
          },
          {
            role: "tool",
            content: [
              {
                type: "tool-result",
                toolCallId: "call_1",
                toolName: "getWeather",
                output: { type: "json", value: { temp: 25 } },
              },
            ],
          },
        ],
      });

      const request = mockCreateChatCompletion.mock.calls[0][0];
      expect(request.messages).toEqual([
        { role: "user", content: "Weather?" },
        {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_1",
              type: "function",
              function: { name: "getWeather", arguments: '{"city":"Tokyo"}' },
            },
          ],
        },
        { role: "tool", tool_call_id: "call_1", content: '{"temp":25}' },
      ]);
    });

    it("uses json_schema response format for generateObject", async () => {
      mockCreateChatCompletion.mockResolvedValue(
        chatResponse({ content: '{"name":"Ada","age":36}' }),
      );
      const model = new WllamaLanguageModel(MODEL_ID);

      const result = await generateObject({
        model,
        schema: z.object({ name: z.string(), age: z.number() }),
        prompt: "Describe a person",
      });

      expect(result.object).toEqual({ name: "Ada", age: 36 });
      const request = mockCreateChatCompletion.mock.calls[0][0];
      expect(request.response_format.type).toBe("json_schema");
      expect(request.response_format.json_schema.schema).toMatchObject({
        type: "object",
        properties: { name: { type: "string" }, age: { type: "number" } },
      });
    });

    it("warns about unsupported stopSequences", async () => {
      mockCreateChatCompletion.mockResolvedValue(chatResponse());
      const model = new WllamaLanguageModel(MODEL_ID);
      const result = await model.doGenerate({
        prompt: userPrompt("hi"),
        stopSequences: ["END"],
      });
      expect(result.warnings).toEqual([
        expect.objectContaining({
          type: "unsupported",
          feature: "stopSequences",
        }),
      ]);
    });

    it("maps length finish reason", async () => {
      mockCreateChatCompletion.mockResolvedValue(
        chatResponse({ finish_reason: "length" }),
      );
      const model = new WllamaLanguageModel(MODEL_ID);
      const result = await model.doGenerate({ prompt: userPrompt("hi") });
      expect(result.finishReason).toEqual({ unified: "length", raw: "length" });
    });

    it("propagates generation errors", async () => {
      mockCreateChatCompletion.mockRejectedValue(new Error("kv_cache_full"));
      const model = new WllamaLanguageModel(MODEL_ID);
      await expect(
        model.doGenerate({ prompt: userPrompt("hi") }),
      ).rejects.toThrow("kv_cache_full");
    });

    it("throws InvalidPromptError when the prompt exceeds the context size", async () => {
      const cause = new Error(CONTEXT_SIZE_MESSAGE);
      mockCreateChatCompletion.mockRejectedValue(cause);
      const model = new WllamaLanguageModel(MODEL_ID);
      const prompt = userPrompt("hi");

      const error = await model.doGenerate({ prompt }).catch((e) => e);

      expect(InvalidPromptError.isInstance(error)).toBe(true);
      expect(error.message).toContain(CONTEXT_SIZE_MESSAGE);
      expect(error.message).toContain("loadParams.n_ctx");
      expect(error.prompt).toBe(prompt);
      expect(error.cause).toBe(cause);
    });
  });

  describe("doStream", () => {
    it("streams text via streamText", async () => {
      mockCreateChatCompletion.mockResolvedValue(
        chunks([
          { content: "Hel" },
          { content: "lo" },
          {
            finish_reason: "stop",
            usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
          },
        ]),
      );
      const model = new WllamaLanguageModel(MODEL_ID);

      const result = streamText({ model, prompt: "Say hi" });
      const text: string[] = [];
      for await (const delta of result.textStream) text.push(delta);

      expect(text.join("")).toBe("Hello");
      expect(await result.finishReason).toBe("stop");
      expect((await result.usage).inputTokens).toBe(3);
      expect(mockCreateChatCompletion.mock.calls[0][0].stream).toBe(true);
    });

    it("accumulates streamed tool call deltas", async () => {
      mockCreateChatCompletion.mockResolvedValue(
        chunks([
          { content: "Let me check." },
          {
            tool_calls: [
              {
                index: 0,
                id: "call_1",
                function: { name: "getWeather", arguments: '{"ci' },
              },
            ],
          },
          {
            tool_calls: [{ index: 0, function: { arguments: 'ty":"Tokyo"}' } }],
          },
          { finish_reason: "tool_calls" },
        ]),
      );
      const model = new WllamaLanguageModel(MODEL_ID);

      const { stream } = await model.doStream({
        prompt: userPrompt("Weather?"),
        tools: [
          {
            type: "function",
            name: "getWeather",
            inputSchema: { type: "object", properties: {} },
          },
        ],
      });

      const parts: any[] = [];
      const reader = stream.getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        parts.push(value);
      }

      expect(parts.map((p) => p.type)).toEqual([
        "stream-start",
        "text-start",
        "text-delta",
        "text-end",
        "tool-input-start",
        "tool-input-delta",
        "tool-input-end",
        "tool-call",
        "finish",
      ]);
      expect(parts.find((p) => p.type === "tool-call")).toEqual({
        type: "tool-call",
        toolCallId: "call_1",
        toolName: "getWeather",
        input: '{"city":"Tokyo"}',
      });
      expect(parts[parts.length - 1].finishReason).toEqual({
        unified: "tool-calls",
        raw: "tool_calls",
      });
    });

    it("emits tool calls delivered in a single chunk", async () => {
      mockCreateChatCompletion.mockResolvedValue(
        chunks([
          {
            tool_calls: [
              {
                index: 0,
                id: "call_1",
                function: { name: "getTime", arguments: "{}" },
              },
            ],
            finish_reason: "tool_calls",
          },
        ]),
      );
      const model = new WllamaLanguageModel(MODEL_ID);
      const { stream } = await model.doStream({ prompt: userPrompt("time?") });

      const parts: any[] = [];
      for await (const part of stream as any) parts.push(part);

      expect(parts.map((p) => p.type)).toEqual([
        "stream-start",
        "tool-input-start",
        "tool-input-delta",
        "tool-input-end",
        "tool-call",
        "finish",
      ]);
    });

    it("finishes with 'other' on abort", async () => {
      mockCreateChatCompletion.mockImplementation(async () => {
        const error = new Error("aborted");
        error.name = "AbortError";
        throw error;
      });
      const model = new WllamaLanguageModel(MODEL_ID);
      const { stream } = await model.doStream({ prompt: userPrompt("hi") });

      const parts: any[] = [];
      for await (const part of stream as any) parts.push(part);

      expect(parts[parts.length - 1]).toMatchObject({
        type: "finish",
        finishReason: { unified: "other", raw: "abort" },
      });
    });

    it("errors the stream with InvalidPromptError when the prompt exceeds the context size", async () => {
      mockCreateChatCompletion.mockRejectedValue(
        new Error(CONTEXT_SIZE_MESSAGE),
      );
      const model = new WllamaLanguageModel(MODEL_ID);
      const { stream } = await model.doStream({ prompt: userPrompt("hi") });

      const error = await (async () => {
        for await (const _ of stream as any) {
          // drain until the stream errors
        }
      })().catch((e) => e);
      expect(InvalidPromptError.isInstance(error)).toBe(true);
      expect(error.message).toContain(CONTEXT_SIZE_MESSAGE);
    });

    it("passes the abort signal through", async () => {
      mockCreateChatCompletion.mockResolvedValue(chunks([{ content: "x" }]));
      const controller = new AbortController();
      const model = new WllamaLanguageModel(MODEL_ID);
      const { stream } = await model.doStream({
        prompt: userPrompt("hi"),
        abortSignal: controller.signal,
      });
      for await (const _ of stream as any) {
        // drain
      }
      expect(mockCreateChatCompletion.mock.calls[0][0].abortSignal).toBe(
        controller.signal,
      );
    });
  });
});
