import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { LiteRTLMLanguageModel, liteRTLM } from "../src";
import { LoadSettingError } from "@ai-sdk/provider";

const mockSendMessage = vi.fn();
const mockSendMessageStreaming = vi.fn();
const mockCancel = vi.fn();
const mockCreateConversation = vi.fn();
const mockEngineCreate = vi.fn();

vi.mock("@litert-lm/core", () => ({
  Engine: {
    create: (...args: unknown[]) => {
      mockEngineCreate(...args);
      return Promise.resolve({
        createConversation: (config: unknown) => {
          mockCreateConversation(config);
          return Promise.resolve({
            sendMessage: (msg: unknown) => {
              mockSendMessage(msg);
              return Promise.resolve({
                content: [{ type: "text", text: "Hello from Gemma" }],
              });
            },
            sendMessageStreaming: (text: unknown) => {
              mockSendMessageStreaming(text);
              return (async function* () {
                yield { content: [{ type: "text", text: "Hello " }] };
                yield { content: [{ type: "text", text: "from Gemma" }] };
              })();
            },
            cancel: mockCancel,
          });
        },
        delete: vi.fn(),
      });
    },
  },
}));

describe("LiteRTLMLanguageModel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    Object.defineProperty(global.navigator, "gpu", {
      value: {},
      configurable: true,
    });
  });

  afterEach(() => {
    Object.defineProperty(global.navigator, "gpu", {
      value: undefined,
      configurable: true,
    });
  });

  describe("constructor", () => {
    it("should create a LiteRTLMLanguageModel instance", () => {
      const model = new LiteRTLMLanguageModel("gemma-4-E2B-it-web.litertlm");
      expect(model).toBeInstanceOf(LiteRTLMLanguageModel);
      expect(model.provider).toBe("litert-lm");
      expect(model.modelId).toBe("gemma-4-E2B-it-web.litertlm");
      expect(model.specificationVersion).toBe("v4");
    });

    it("should expose the default provider singleton", () => {
      const model = liteRTLM("gemma-4-E2B-it-web.litertlm");
      expect(model).toBeInstanceOf(LiteRTLMLanguageModel);
    });
  });

  describe("availability", () => {
    it("should report unavailable without WebGPU", async () => {
      Object.defineProperty(global.navigator, "gpu", {
        value: undefined,
        configurable: true,
      });
      const model = new LiteRTLMLanguageModel("m.litertlm");
      expect(await model.availability()).toBe("unavailable");
    });

    it("should report downloadable with WebGPU", async () => {
      const model = new LiteRTLMLanguageModel("m.litertlm");
      expect(await model.availability()).toBe("downloadable");
    });
  });

  describe("doGenerate", () => {
    it("should generate text from a prompt", async () => {
      const model = new LiteRTLMLanguageModel("m.litertlm");
      const result = await model.doGenerate({
        prompt: [
          { role: "system", content: "You are helpful." },
          {
            role: "user",
            content: [{ type: "text", text: "Hi!" }],
          },
        ],
        abortSignal: undefined as unknown as AbortSignal,
      });

      expect(result.content).toEqual([
        { type: "text", text: "Hello from Gemma" },
      ]);
      expect(result.finishReason).toEqual({ unified: "stop", raw: "stop" });

      // History (system + prior user) is passed as the conversation preface;
      // only the final user message is sent.
      expect(mockCreateConversation).toHaveBeenCalledWith(
        expect.objectContaining({
          preface: { messages: [{ role: "system", content: "You are helpful." }] },
        }),
      );
      expect(mockSendMessage).toHaveBeenCalledWith({
        role: "user",
        content: "Hi!",
      });
    });

    it("should throw LoadSettingError when WebGPU is unavailable", async () => {
      Object.defineProperty(global.navigator, "gpu", {
        value: undefined,
        configurable: true,
      });
      const model = new LiteRTLMLanguageModel("m.litertlm");
      await expect(
        model.doGenerate({
          prompt: [{ role: "user", content: [{ type: "text", text: "Hi" }] }],
          abortSignal: undefined as unknown as AbortSignal,
        }),
      ).rejects.toThrow(LoadSettingError);
    });
  });

  describe("doStream", () => {
    it("should stream text deltas", async () => {
      const model = new LiteRTLMLanguageModel("m.litertlm");
      const result = await model.doStream({
        prompt: [{ role: "user", content: [{ type: "text", text: "Hi" }] }],
        abortSignal: undefined as unknown as AbortSignal,
      });

      const parts: unknown[] = [];
      for await (const part of result.stream) {
        parts.push(part);
      }

      const textDeltas = parts.filter((p) => (p as any).type === "text-delta");
      expect(textDeltas).toEqual([
        { type: "text-delta", id: "text-0", delta: "Hello " },
        { type: "text-delta", id: "text-0", delta: "from Gemma" },
      ]);
      expect(parts.some((p) => (p as any).type === "finish")).toBe(true);
      expect(mockSendMessageStreaming).toHaveBeenCalledWith("Hi");
    });
  });
});
