import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@huggingface/transformers", () => {
  const tokenizer = vi.fn().mockReturnValue({
    input_ids: { data: Int32Array.from([101]) },
  });
  (tokenizer as any).apply_chat_template = vi.fn();
  (tokenizer as any).decode = vi
    .fn()
    .mockImplementation((tokens: number[]) => tokens.join(","));

  const model = {
    generate: vi.fn(),
  };

  class TextStreamer {
    constructor(_tokenizer: unknown, _options?: unknown) {
      // no-op
    }
  }

  class InterruptableStoppingCriteria {
    interrupted = false;
    interrupt() {
      this.interrupted = true;
    }
    reset() {
      this.interrupted = false;
    }
    _call() {
      return [this.interrupted];
    }
  }

  class StoppingCriteriaList {
    private items: unknown[] = [];
    push(item: unknown) {
      this.items.push(item);
    }
  }

  return {
    AutoTokenizer: { from_pretrained: vi.fn().mockResolvedValue(tokenizer) },
    AutoModelForCausalLM: { from_pretrained: vi.fn().mockResolvedValue(model) },
    AutoProcessor: { from_pretrained: vi.fn() },
    AutoModelForVision2Seq: { from_pretrained: vi.fn() },
    TextStreamer,
    InterruptableStoppingCriteria,
    StoppingCriteriaList,
    load_image: vi.fn(),
    __TEST_MOCK__: { tokenizer, model },
  };
});

import { TransformersJSWorkerHandler } from "../src/chat/transformers-js-worker-handler";

describe("TransformersJSWorkerHandler", () => {
  let tokenizerMock: any;
  let modelMock: any;
  let postMessageMock: any;

  beforeEach(async () => {
    vi.clearAllMocks();
    postMessageMock = vi.fn();
    (globalThis as any).self = {
      postMessage: postMessageMock,
      addEventListener: vi.fn(),
    };

    const hf = await import("@huggingface/transformers");
    tokenizerMock = (hf as any).__TEST_MOCK__.tokenizer;
    modelMock = (hf as any).__TEST_MOCK__.model;
    tokenizerMock.apply_chat_template.mockReset();
    tokenizerMock.decode.mockImplementation((tokens: number[]) =>
      tokens.join(","),
    );
    modelMock.generate.mockReset();
  });

  it("reuses past_key_values when prompt extends previous generation", async () => {
    const handler = new TransformersJSWorkerHandler();

    // load() performs one warmup generate call for text models
    modelMock.generate.mockResolvedValueOnce({
      sequences: [{ data: Int32Array.from([101, 102]) }],
    });
    await handler.load({ modelId: "test-model" });

    modelMock.generate.mockReset();
    tokenizerMock.apply_chat_template
      .mockReturnValueOnce({ input_ids: { data: Int32Array.from([1, 2, 3]) } })
      .mockReturnValueOnce({
        input_ids: { data: Int32Array.from([1, 2, 3, 10, 4]) },
      });

    modelMock.generate
      .mockResolvedValueOnce({
        sequences: [{ data: Int32Array.from([1, 2, 3, 10]) }],
        past_key_values: { cache: "k1" },
      })
      .mockResolvedValueOnce({
        sequences: [{ data: Int32Array.from([1, 2, 3, 10, 4, 11]) }],
        past_key_values: { cache: "k2" },
      });

    await handler.generate([{ role: "user", content: "hi" }]);
    await handler.generate([
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello" },
      { role: "user", content: "next" },
    ]);

    expect(modelMock.generate).toHaveBeenCalledTimes(2);
    expect(modelMock.generate.mock.calls[0][0].past_key_values).toBeUndefined();
    expect(modelMock.generate.mock.calls[1][0].past_key_values).toEqual({
      cache: "k1",
    });
  });

  it("retries once without past_key_values when cached prefill fails", async () => {
    const handler = new TransformersJSWorkerHandler();

    // load() warmup
    modelMock.generate.mockResolvedValueOnce({
      sequences: [{ data: Int32Array.from([101, 102]) }],
    });
    await handler.load({ modelId: "test-model" });

    modelMock.generate.mockReset();
    tokenizerMock.apply_chat_template
      .mockReturnValueOnce({ input_ids: { data: Int32Array.from([1, 2, 3]) } })
      .mockReturnValueOnce({
        input_ids: { data: Int32Array.from([1, 2, 3, 10, 4]) },
      });

    modelMock.generate
      .mockResolvedValueOnce({
        sequences: [{ data: Int32Array.from([1, 2, 3, 10]) }],
        past_key_values: { cache: "k1" },
      })
      .mockImplementationOnce((args: any) => {
        if (args.past_key_values) {
          throw new Error("kv mismatch");
        }
        return {
          sequences: [{ data: Int32Array.from([1, 2, 3, 10, 4, 11]) }],
          past_key_values: { cache: "k2" },
        };
      })
      .mockResolvedValueOnce({
        sequences: [{ data: Int32Array.from([1, 2, 3, 10, 4, 11]) }],
        past_key_values: { cache: "k2" },
      });

    await handler.generate([{ role: "user", content: "hi" }]);
    await handler.generate([
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello" },
      { role: "user", content: "next" },
    ]);

    expect(modelMock.generate).toHaveBeenCalledTimes(3);
    expect(modelMock.generate.mock.calls[1][0].past_key_values).toEqual({
      cache: "k1",
    });
    expect(modelMock.generate.mock.calls[2][0].past_key_values).toBeUndefined();
    expect(
      postMessageMock.mock.calls.some(
        (call: any[]) => call[0]?.status === "error",
      ),
    ).toBe(false);
  });
});
