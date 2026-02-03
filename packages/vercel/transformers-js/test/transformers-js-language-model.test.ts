import { describe, it, expect, vi, beforeEach } from "vitest";
import { generateText, streamText, generateObject } from "ai";
import { z } from "zod";

vi.mock("@huggingface/transformers", () => {
  // Create a tokenizer function that also has methods
  const tokenizer = vi.fn().mockReturnValue({ input_ids: [1, 2, 3] });
  (tokenizer as any).apply_chat_template = vi.fn();
  (tokenizer as any).decode = vi.fn();

  const model = {
    generate: vi.fn().mockResolvedValue([1, 2, 3]),
  };
  class TextStreamer {
    private callback: ((text: string) => void) | undefined;
    constructor(
      _tokenizer: any,
      options?: { callback_function?: (text: string) => void },
    ) {
      this.callback = options?.callback_function;
    }
    on_finalized_text(text: string): void {
      this.callback?.(text);
    }
  }
  class StoppingCriteria {
    _call() {
      return [false];
    }
  }
  class StoppingCriteriaList {
    extend(_arr: any[]) {
      /* no-op */
    }
  }
  // Mock env object for testing localModelPath and cacheDir
  const env = {
    localModelPath: undefined as string | undefined,
    cacheDir: undefined as string | undefined,
  };
  return {
    AutoTokenizer: { from_pretrained: vi.fn().mockResolvedValue(tokenizer) },
    AutoModelForCausalLM: { from_pretrained: vi.fn().mockResolvedValue(model) },
    TextStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
    env,
    __TEST_MOCK__: { tokenizer, model, env },
  };
});

import { TransformersJSLanguageModel } from "../src";

describe("TransformersJSLanguageModel", () => {
  let tokenizerMock: any;
  let modelMock: any;
  let envMock: any;
  beforeEach(() => {
    vi.clearAllMocks();
    // Load mocked module to access test doubles
    return import("@huggingface/transformers").then((m: any) => {
      tokenizerMock = m.__TEST_MOCK__.tokenizer;
      modelMock = m.__TEST_MOCK__.model;
      envMock = m.__TEST_MOCK__.env;
      // Reset env mock values
      envMock.localModelPath = undefined;
      envMock.cacheDir = undefined;
    });
  });

  it("instantiates and reports downloadable before init", async () => {
    const model = new TransformersJSLanguageModel(
      "HuggingFaceTB/SmolLM2-360M-Instruct",
    );
    const availability = await model.availability();
    expect(availability).toBe("downloadable");
  });

  it("generate returns text and usage", async () => {
    const model = new TransformersJSLanguageModel(
      "HuggingFaceTB/SmolLM2-360M-Instruct",
    );

    // tokenizer returns tensors/arrays the class expects
    tokenizerMock.apply_chat_template.mockReturnValue({
      input_ids: { data: new Array(5).fill(1) },
    });

    // mock model.generate to call streamer callback (unified streaming approach)
    (modelMock.generate as any).mockImplementation(async (args: any) => {
      if (args.streamer) {
        args.streamer.on_finalized_text("Hello");
      }
      return Promise.resolve();
    });

    const { text, usage } = await generateText({
      model,
      prompt: "Say hello",
    });

    expect(text).toBe("Hello");
    expect(usage).toMatchObject({
      inputTokens: 5,
      outputTokens: 1,
      totalTokens: 6,
    });
  });

  it("reports correct availability", async () => {
    const model = new TransformersJSLanguageModel(
      "HuggingFaceTB/SmolLM2-360M-Instruct",
    );
    const availability = await model.availability();
    expect(availability).toBe("downloadable");
  });

  it("should handle system messages", async () => {
    const model = new TransformersJSLanguageModel(
      "HuggingFaceTB/SmolLM2-360M-Instruct",
    );

    tokenizerMock.apply_chat_template.mockReturnValue({
      input_ids: { data: new Array(3).fill(1) },
    });
    // mock model.generate to call streamer callback (unified streaming approach)
    (modelMock.generate as any).mockImplementation(async (args: any) => {
      if (args.streamer) {
        args.streamer.on_finalized_text("I am a helpful assistant.");
      }
      return Promise.resolve();
    });

    const { text } = await generateText({
      model,
      messages: [
        { role: "system", content: "You are a helpful assistant." },
        { role: "user", content: "Who are you?" },
      ],
    });

    expect(text).toBe("I am a helpful assistant.");
    // System message should be passed through natively
    const applyChatCall = tokenizerMock.apply_chat_template.mock.calls[0];
    expect(applyChatCall[0]).toHaveLength(2);
    expect(applyChatCall[0][0].role).toBe("system");
    expect(applyChatCall[0][0].content).toBe("You are a helpful assistant.");
    expect(applyChatCall[0][1].role).toBe("user");
    expect(applyChatCall[0][1].content).toBe("Who are you?");
  });

  it("should handle conversation history", async () => {
    const model = new TransformersJSLanguageModel(
      "HuggingFaceTB/SmolLM2-360M-Instruct",
    );

    tokenizerMock.apply_chat_template.mockReturnValue({
      input_ids: { data: new Array(4).fill(1) },
    });
    // mock model.generate to call streamer callback (unified streaming approach)
    (modelMock.generate as any).mockImplementation(async (args: any) => {
      if (args.streamer) {
        args.streamer.on_finalized_text("I can help you with coding!");
      }
      return Promise.resolve();
    });

    const { text } = await generateText({
      model,
      messages: [
        { role: "user", content: "Can you help me?" },
        { role: "assistant", content: "Of course! What do you need?" },
        { role: "user", content: "I need assistance with coding." },
      ],
    });

    expect(text).toBe("I can help you with coding!");
    // Should receive full conversation history
    const applyChatCall = tokenizerMock.apply_chat_template.mock.calls[0];
    expect(applyChatCall[0]).toEqual([
      { role: "user", content: "Can you help me?" },
      { role: "assistant", content: "Of course! What do you need?" },
      { role: "user", content: "I need assistance with coding." },
    ]);
  });

  it("should stream text successfully", async () => {
    const model = new TransformersJSLanguageModel(
      "HuggingFaceTB/SmolLM2-360M-Instruct",
    );

    tokenizerMock.apply_chat_template.mockReturnValue({
      input_ids: { data: new Array(2).fill(1) },
    });

    // Mock the generate method to simulate streaming by calling the streamer callback
    (modelMock.generate as any).mockImplementation(async (args: any) => {
      // Simulate streamer callbacks synchronously
      if (args.streamer) {
        args.streamer.on_finalized_text("Hello");
        args.streamer.on_finalized_text(", ");
        args.streamer.on_finalized_text("world!");
      }
      return Promise.resolve();
    });

    const { textStream, usage } = streamText({
      model,
      prompt: "Say hello",
    });

    let acc = "";
    for await (const chunk of textStream) {
      acc += chunk;
    }

    expect(acc).toBe("Hello, world!");
    const usageResult = await usage;
    expect(usageResult.inputTokens).toBe(2);
    expect(usageResult.outputTokens).toBeGreaterThanOrEqual(1);
  });

  it("should handle empty content arrays", async () => {
    const model = new TransformersJSLanguageModel(
      "HuggingFaceTB/SmolLM2-360M-Instruct",
    );

    tokenizerMock.apply_chat_template.mockReturnValue({
      input_ids: { data: new Array(1).fill(1) },
    });
    // mock model.generate to call streamer callback (unified streaming approach)
    (modelMock.generate as any).mockImplementation(async (args: any) => {
      if (args.streamer) {
        args.streamer.on_finalized_text("Response");
      }
      return Promise.resolve();
    });

    const { text } = await generateText({
      model,
      messages: [
        {
          role: "user",
          content: [],
        },
      ],
    });

    expect(text).toBe("Response");
    // Should pass empty content to chat template
    const applyChatCall = tokenizerMock.apply_chat_template.mock.calls[0];
    expect(applyChatCall[0]).toEqual([{ role: "user", content: "" }]);
  });

  it("should configure env.localModelPath when provided", async () => {
    const model = new TransformersJSLanguageModel(
      "HuggingFaceTB/SmolLM2-360M-Instruct",
      { localModelPath: "/custom/models" },
    );

    tokenizerMock.apply_chat_template.mockReturnValue({
      input_ids: { data: new Array(2).fill(1) },
    });
    (modelMock.generate as any).mockImplementation(async (args: any) => {
      if (args.streamer) {
        args.streamer.on_finalized_text("Hello");
      }
      return Promise.resolve();
    });

    await model.createSessionWithProgress();

    expect(envMock.localModelPath).toBe("/custom/models");
  });

  it("should configure env.cacheDir when provided", async () => {
    const model = new TransformersJSLanguageModel(
      "HuggingFaceTB/SmolLM2-360M-Instruct",
      { cacheDir: "/custom/cache" },
    );

    tokenizerMock.apply_chat_template.mockReturnValue({
      input_ids: { data: new Array(2).fill(1) },
    });
    (modelMock.generate as any).mockImplementation(async (args: any) => {
      if (args.streamer) {
        args.streamer.on_finalized_text("Hello");
      }
      return Promise.resolve();
    });

    await model.createSessionWithProgress();

    expect(envMock.cacheDir).toBe("/custom/cache");
  });

  it("should not modify env when localModelPath and cacheDir are not provided", async () => {
    const model = new TransformersJSLanguageModel(
      "HuggingFaceTB/SmolLM2-360M-Instruct",
    );

    tokenizerMock.apply_chat_template.mockReturnValue({
      input_ids: { data: new Array(2).fill(1) },
    });
    (modelMock.generate as any).mockImplementation(async (args: any) => {
      if (args.streamer) {
        args.streamer.on_finalized_text("Hello");
      }
      return Promise.resolve();
    });

    await model.createSessionWithProgress();

    expect(envMock.localModelPath).toBeUndefined();
    expect(envMock.cacheDir).toBeUndefined();
  });
});
