import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { WebLLMLanguageModel, webLLM } from "../src";
import { generateText, streamText } from "ai";
import { LoadSettingError } from "@ai-sdk/provider";

// Mock the external dependency
const mockChatCompletionsCreate = vi.fn();
const mockInterruptGenerate = vi.fn();
const mockReload = vi.fn();
const mockEngineConstructor = vi.fn();

vi.mock("@mlc-ai/web-llm", () => ({
  MLCEngine: vi.fn().mockImplementation(function (config) {
    mockEngineConstructor(config);
    return {
      chat: {
        completions: {
          create: mockChatCompletionsCreate,
        },
      },
      reload: mockReload,
      interruptGenerate: mockInterruptGenerate,
    };
  }),
  CreateWebWorkerMLCEngine: vi.fn().mockImplementation(() =>
    Promise.resolve({
      chat: {
        completions: {
          create: mockChatCompletionsCreate,
        },
      },
      interruptGenerate: mockInterruptGenerate,
    }),
  ),
}));

describe("WebLLMLanguageModel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Mock navigator.gpu
    Object.defineProperty(global.navigator, "gpu", {
      value: {},
      configurable: true,
    });
  });

  afterEach(() => {
    // Restore original navigator.gpu
    Object.defineProperty(global.navigator, "gpu", {
      value: undefined,
      configurable: true,
    });
  });

  describe("constructor", () => {
    it("should create a WebLLMLanguageModel instance", () => {
      const model = new WebLLMLanguageModel(
        "Llama-3.1-8B-Instruct-q4f32_1-MLC",
      );

      expect(model).toBeInstanceOf(WebLLMLanguageModel);
      expect(model.specificationVersion).toBe("v3");
      expect(model.provider).toBe("web-llm");
      expect(model.modelId).toBe("Llama-3.1-8B-Instruct-q4f32_1-MLC");
    });

    it("should create a model using the factory function", () => {
      const model = webLLM("Llama-3.1-8B-Instruct-q4f32_1-MLC");

      expect(model).toBeInstanceOf(WebLLMLanguageModel);
      expect(model.modelId).toBe("Llama-3.1-8B-Instruct-q4f32_1-MLC");
    });

    it("should accept settings in the factory function", () => {
      const model = webLLM("Llama-3.1-8B-Instruct-q4f32_1-MLC", {
        initProgressCallback: (progress) => console.log(progress),
      });

      expect(model).toBeInstanceOf(WebLLMLanguageModel);
    });
  });

  describe("availability", () => {
    it("should return 'unavailable' if navigator.gpu is not supported", async () => {
      Object.defineProperty(global.navigator, "gpu", {
        value: undefined,
        configurable: true,
      });
      const model = new WebLLMLanguageModel("test-model");
      const availability = await model.availability();
      expect(availability).toBe("unavailable");
    });

    it("should return 'downloadable' if not initialized", async () => {
      const model = new WebLLMLanguageModel("test-model");
      const availability = await model.availability();
      expect(availability).toBe("downloadable");
    });

    it("should return 'available' if initialized", async () => {
      const model = new WebLLMLanguageModel("test-model");
      // Manually set as initialized for test purposes
      (model as any).isInitialized = true;
      const availability = await model.availability();
      expect(availability).toBe("available");
    });
  });

  describe("doGenerate", () => {
    it("should throw LoadSettingError if WebLLM is not supported", async () => {
      Object.defineProperty(global.navigator, "gpu", {
        value: undefined,
        configurable: true,
      });
      const model = new WebLLMLanguageModel("test-model");
      await expect(
        model.doGenerate({
          prompt: [
            { role: "user", content: [{ type: "text", text: "hello" }] },
          ],
        }),
      ).rejects.toThrow(
        new LoadSettingError({
          message:
            "WebLLM is not available. This library requires a browser with WebGPU support.",
        }),
      );
    });

    it("should generate text successfully", async () => {
      mockChatCompletionsCreate.mockResolvedValue({
        choices: [
          {
            message: { content: "Hello, world!" },
            finish_reason: "stop",
          },
        ],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 5,
          total_tokens: 15,
        },
      });

      const model = new WebLLMLanguageModel("test-model");
      const { text, usage } = await generateText({
        model,
        prompt: "Say hello",
      });

      expect(text).toBe("Hello, world!");
      expect(usage).toMatchObject({
        inputTokens: 10,
        outputTokens: 5,
        totalTokens: 15,
      });
      expect(mockReload).toHaveBeenCalledWith("test-model");
      expect(mockChatCompletionsCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          messages: [{ role: "user", content: "Say hello" }],
          stream: false,
        }),
      );
    });
  });

  describe("doStream", () => {
    it("should stream text successfully", async () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      async function* createAsyncGenerator(): AsyncGenerator<
        any,
        void,
        unknown
      > {
        yield {
          choices: [{ delta: { content: "Hello" } }],
        };
        yield {
          choices: [{ delta: { content: ", " } }],
        };
        yield {
          choices: [{ delta: { content: "world!" } }],
        };
        yield {
          choices: [
            {
              delta: {},
              finish_reason: "stop",
            },
          ],
          usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
        };
      }

      mockChatCompletionsCreate.mockResolvedValue(createAsyncGenerator());

      const model = new WebLLMLanguageModel("test-model");
      const { textStream, usage } = await streamText({
        model,
        prompt: "Say hello",
      });

      let text = "";
      for await (const chunk of textStream) {
        text += chunk;
      }

      expect(text).toBe("Hello, world!");
      const usageResult = await usage;
      expect(usageResult).toMatchObject({
        inputTokens: 10,
        outputTokens: 5,
        totalTokens: 15,
      });

      expect(mockChatCompletionsCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          messages: [{ role: "user", content: "Say hello" }],
          stream: true,
        }),
      );
    });
  });

  describe("createSessionWithProgress", () => {
    it("should call onDownloadProgress with unified progress during initialization", async () => {
      mockEngineConstructor.mockImplementation((config) => {
        if (config.initProgressCallback) {
          config.initProgressCallback({ progress: 0.5, text: "loading" });
        }
      });

      const model = new WebLLMLanguageModel("test-model");
      const onDownloadProgress = vi.fn();

      const result = await model.createSessionWithProgress(onDownloadProgress);

      expect(onDownloadProgress).toHaveBeenCalledWith(0.5);
      expect(result).toBe(model);
    });
  });

  describe("tool calling", () => {
    describe("doGenerate with tools", () => {
      it("should detect tool calls and return tool-calls finish reason", async () => {
        const toolCallResponse = `\`\`\`tool_call
{"name": "get_weather", "arguments": {"city": "San Francisco"}}
\`\`\``;

        mockChatCompletionsCreate.mockResolvedValue({
          choices: [
            {
              message: { content: toolCallResponse },
              finish_reason: "stop",
            },
          ],
          usage: {
            prompt_tokens: 50,
            completion_tokens: 20,
            total_tokens: 70,
          },
        });

        const model = new WebLLMLanguageModel("test-model");
        const result = await model.doGenerate({
          prompt: [
            {
              role: "user",
              content: [
                { type: "text", text: "What's the weather in San Francisco?" },
              ],
            },
          ],
          tools: [
            {
              type: "function",
              name: "get_weather",
              description: "Get the weather for a city",
              inputSchema: {
                type: "object",
                properties: {
                  city: { type: "string" },
                },
                required: ["city"],
              },
            },
          ],
        });

        expect(result.finishReason).toMatchObject({
          unified: "tool-calls",
          raw: "tool-calls",
        });
        expect(result.content).toHaveLength(1);
        expect(result.content[0].type).toBe("tool-call");

        const toolCall = result.content[0];
        if (toolCall.type === "tool-call") {
          expect(toolCall.toolName).toBe("get_weather");
          expect(JSON.parse(toolCall.input)).toEqual({
            city: "San Francisco",
          });
          expect(toolCall.toolCallId).toMatch(/^call_/);
        }
      });

      it("should handle tool calls with preceding text", async () => {
        const response = `Let me check the weather for you.
\`\`\`tool_call
{"name": "get_weather", "arguments": {"city": "NYC"}}
\`\`\``;

        mockChatCompletionsCreate.mockResolvedValue({
          choices: [
            {
              message: { content: response },
              finish_reason: "stop",
            },
          ],
          usage: {
            prompt_tokens: 40,
            completion_tokens: 25,
            total_tokens: 65,
          },
        });

        const model = new WebLLMLanguageModel("test-model");
        const result = await model.doGenerate({
          prompt: [
            {
              role: "user",
              content: [{ type: "text", text: "What's the weather in NYC?" }],
            },
          ],
          tools: [
            {
              type: "function",
              name: "get_weather",
              description: "Get the weather for a city",
              inputSchema: {
                type: "object",
                properties: {
                  city: { type: "string" },
                },
              },
            },
          ],
        });

        expect(result.finishReason).toMatchObject({
          unified: "tool-calls",
          raw: "tool-calls",
        });
        expect(result.content).toHaveLength(2);

        expect(result.content[0].type).toBe("text");
        if (result.content[0].type === "text") {
          expect(result.content[0].text).toContain("Let me check the weather");
        }

        expect(result.content[1].type).toBe("tool-call");
      });

      it("should return normal text when no tool calls detected", async () => {
        mockChatCompletionsCreate.mockResolvedValue({
          choices: [
            {
              message: { content: "I don't need to use any tools for this." },
              finish_reason: "stop",
            },
          ],
          usage: {
            prompt_tokens: 30,
            completion_tokens: 10,
            total_tokens: 40,
          },
        });

        const model = new WebLLMLanguageModel("test-model");
        const result = await model.doGenerate({
          prompt: [
            {
              role: "user",
              content: [{ type: "text", text: "Say hello" }],
            },
          ],
          tools: [
            {
              type: "function",
              name: "get_weather",
              description: "Get the weather",
              inputSchema: {
                type: "object",
                properties: {},
              },
            },
          ],
        });

        expect(result.finishReason).toMatchObject({
          unified: "stop",
          raw: "stop",
        });
        expect(result.content).toHaveLength(1);
        expect(result.content[0].type).toBe("text");
      });

      it("should only emit first tool call when multiple are present", async () => {
        const response = `\`\`\`tool_call
[
  {"name": "tool1", "arguments": {"a": 1}},
  {"name": "tool2", "arguments": {"b": 2}}
]
\`\`\``;

        mockChatCompletionsCreate.mockResolvedValue({
          choices: [
            {
              message: { content: response },
              finish_reason: "stop",
            },
          ],
          usage: {
            prompt_tokens: 40,
            completion_tokens: 30,
            total_tokens: 70,
          },
        });

        const model = new WebLLMLanguageModel("test-model");
        const result = await model.doGenerate({
          prompt: [
            {
              role: "user",
              content: [{ type: "text", text: "Use tools" }],
            },
          ],
          tools: [
            {
              type: "function",
              name: "tool1",
              description: "First tool",
              inputSchema: { type: "object", properties: {} },
            },
            {
              type: "function",
              name: "tool2",
              description: "Second tool",
              inputSchema: { type: "object", properties: {} },
            },
          ],
        });

        expect(result.finishReason).toMatchObject({
          unified: "tool-calls",
          raw: "tool-calls",
        });
        expect(result.content).toHaveLength(1);
        // Should only emit first tool call
        const toolCalls = result.content.filter((c) => c.type === "tool-call");
        expect(toolCalls).toHaveLength(1);

        if (toolCalls[0].type === "tool-call") {
          expect(toolCalls[0].toolName).toBe("tool1");
        }
      });
    });

    describe("doStream with tools", () => {
      it("should stream tool calls in real-time", async () => {
        async function* createToolCallStream(): AsyncGenerator<
          any,
          void,
          unknown
        > {
          yield { choices: [{ delta: { content: "```tool_call\n" } }] };
          yield {
            choices: [
              { delta: { content: '{"name": "get_weather", "arguments": {' } },
            ],
          };
          yield { choices: [{ delta: { content: '"city": "SF"}}' } }] };
          yield { choices: [{ delta: { content: "\n```" } }] };
          yield {
            choices: [{ delta: {}, finish_reason: "stop" }],
            usage: {
              prompt_tokens: 50,
              completion_tokens: 25,
              total_tokens: 75,
            },
          };
        }

        mockChatCompletionsCreate.mockResolvedValue(createToolCallStream());

        const model = new WebLLMLanguageModel("test-model");
        const { stream } = await model.doStream({
          prompt: [
            {
              role: "user",
              content: [{ type: "text", text: "What's the weather in SF?" }],
            },
          ],
          tools: [
            {
              type: "function",
              name: "get_weather",
              description: "Get weather",
              inputSchema: {
                type: "object",
                properties: { city: { type: "string" } },
              },
            },
          ],
        });

        const parts = [];
        const reader = stream.getReader();
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            parts.push(value);
          }
        } finally {
          reader.releaseLock();
        }

        // Should have tool-input-start
        const toolInputStart = parts.find((p) => p.type === "tool-input-start");
        expect(toolInputStart).toBeDefined();
        if (toolInputStart?.type === "tool-input-start") {
          expect(toolInputStart.toolName).toBe("get_weather");
        }

        // Should have tool-call
        const toolCall = parts.find((p) => p.type === "tool-call");
        expect(toolCall).toBeDefined();

        // Should have tool-calls finish reason
        const finish = parts.find((p) => p.type === "finish");
        if (finish?.type === "finish") {
          expect(finish.finishReason).toMatchObject({
            unified: "tool-calls",
            raw: "tool-calls",
          });
        }
      });

      it("should stream text without tools correctly", async () => {
        async function* createTextStream(): AsyncGenerator<any, void, unknown> {
          yield { choices: [{ delta: { content: "Hello" } }] };
          yield { choices: [{ delta: { content: ", " } }] };
          yield { choices: [{ delta: { content: "world!" } }] };
          yield {
            choices: [{ delta: {}, finish_reason: "stop" }],
            usage: {
              prompt_tokens: 10,
              completion_tokens: 5,
              total_tokens: 15,
            },
          };
        }

        mockChatCompletionsCreate.mockResolvedValue(createTextStream());

        const model = new WebLLMLanguageModel("test-model");
        const { stream } = await model.doStream({
          prompt: [
            {
              role: "user",
              content: [{ type: "text", text: "Say hello" }],
            },
          ],
          tools: [
            {
              type: "function",
              name: "get_weather",
              description: "Get weather",
              inputSchema: {
                type: "object",
                properties: {
                  city: { type: "string" },
                },
              },
            },
          ],
        });

        const parts = [];
        const reader = stream.getReader();
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            parts.push(value);
          }
        } finally {
          reader.releaseLock();
        }

        // Collect text deltas
        const textDeltas = parts.filter((p) => p.type === "text-delta");
        const text = textDeltas.map((p) => (p as any).delta).join("");

        expect(text).toBe("Hello, world!");

        const finish = parts.find((p) => p.type === "finish");
        if (finish?.type === "finish") {
          expect(finish.finishReason).toMatchObject({
            unified: "stop",
            raw: "stop",
          });
        }
      });

      it("should handle text before tool call in stream", async () => {
        async function* createMixedStream(): AsyncGenerator<
          any,
          void,
          unknown
        > {
          yield { choices: [{ delta: { content: "Let me help. " } }] };
          yield { choices: [{ delta: { content: "```tool_call\n" } }] };
          yield {
            choices: [{ delta: { content: '{"name": "help"}' } }],
          };
          yield { choices: [{ delta: { content: "\n```" } }] };
          yield {
            choices: [{ delta: {}, finish_reason: "stop" }],
            usage: {
              prompt_tokens: 30,
              completion_tokens: 15,
              total_tokens: 45,
            },
          };
        }

        mockChatCompletionsCreate.mockResolvedValue(createMixedStream());

        const model = new WebLLMLanguageModel("test-model");
        const { stream } = await model.doStream({
          prompt: [
            {
              role: "user",
              content: [{ type: "text", text: "Help me" }],
            },
          ],
          tools: [
            {
              type: "function",
              name: "help",
              description: "Help tool",
              inputSchema: { type: "object", properties: {} },
            },
          ],
        });

        const parts = [];
        const reader = stream.getReader();
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            parts.push(value);
          }
        } finally {
          reader.releaseLock();
        }

        // Should have text parts before tool call
        const textDeltas = parts.filter((p) => p.type === "text-delta");
        expect(textDeltas.length).toBeGreaterThan(0);

        // Should also have tool call
        const toolCall = parts.find((p) => p.type === "tool-call");
        expect(toolCall).toBeDefined();
      });
    });

    it("should emit warnings for unsupported tool settings", async () => {
      mockChatCompletionsCreate.mockResolvedValue({
        choices: [
          {
            message: { content: "Response" },
            finish_reason: "stop",
          },
        ],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 5,
          total_tokens: 15,
        },
      });

      const model = new WebLLMLanguageModel("test-model");
      const result = await model.doGenerate({
        prompt: [
          {
            role: "user",
            content: [{ type: "text", text: "Test" }],
          },
        ],
        tools: [
          {
            type: "function",
            name: "test",
            description: "Test",
            inputSchema: {
              type: "object",
              properties: {
                param: { type: "string" },
              },
            },
          },
        ],
        toolChoice: { type: "tool", toolName: "test" } as any,
      });

      expect(result.warnings).toBeDefined();
      expect(result.warnings?.length).toBeGreaterThan(0);

      const toolChoiceWarning = result.warnings?.find(
        (w) => w.type === "unsupported",
      );
      expect(toolChoiceWarning).toBeDefined();
    });
  });
});
