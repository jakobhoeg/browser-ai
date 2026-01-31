import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import {
  BrowserAIChatLanguageModel,
  BrowserAIChatSettings,
} from "../src/browser-ai-language-model";

import { generateText, streamText, generateObject, streamObject } from "ai";
import { LanguageModelV3StreamPart, LoadSettingError } from "@ai-sdk/provider";
import { z } from "zod";

describe("BrowserAIChatLanguageModel", () => {
  let mockSession: any;
  let mockPrompt: any;
  let mockPromptStreaming: any;

  beforeEach(() => {
    // Reset all mocks
    vi.clearAllMocks();

    // Create mock session
    mockPrompt = vi.fn();
    mockPromptStreaming = vi.fn();
    mockSession = {
      prompt: mockPrompt,
      promptStreaming: mockPromptStreaming,
      destroy: vi.fn(),
      inputUsage: 0,
      addEventListener: vi.fn(),
    };
    // Mock the global LanguageModel API
    vi.stubGlobal("LanguageModel", {
      availability: vi.fn().mockResolvedValue("available"),
      create: vi.fn().mockResolvedValue(mockSession),
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("should instantiate correctly", () => {
    const model = new BrowserAIChatLanguageModel("text");
    expect(model).toBeInstanceOf(BrowserAIChatLanguageModel);
    expect(model.modelId).toBe("text");
    expect(model.provider).toBe("browser-ai");
    expect(model.specificationVersion).toBe("v3");
  });
  it("should throw when LanguageModel is not available", async () => {
    vi.stubGlobal("LanguageModel", undefined);

    await expect(() =>
      generateText({
        model: new BrowserAIChatLanguageModel("text"),
        prompt: "test",
      }),
    ).rejects.toThrow(LoadSettingError);
  });
  it("should throw when model is unavailable", async () => {
    vi.stubGlobal("LanguageModel", {
      availability: vi.fn().mockResolvedValue("unavailable"),
    });

    await expect(() =>
      generateText({
        model: new BrowserAIChatLanguageModel("text"),
        prompt: "test",
      }),
    ).rejects.toThrow(LoadSettingError);
  });

  it("should generate text successfully", async () => {
    mockPrompt.mockResolvedValue("Hello, world!");

    const result = await generateText({
      model: new BrowserAIChatLanguageModel("text"),
      prompt: "Say hello",
    });

    expect(result.text).toBe("Hello, world!");
    expect(mockPrompt).toHaveBeenCalledWith(
      [
        {
          role: "user",
          content: [{ type: "text", value: "Say hello" }],
        },
      ],
      {},
    );
  });

  it("should handle system messages", async () => {
    mockPrompt.mockResolvedValue("I am a helpful assistant.");

    const result = await generateText({
      model: new BrowserAIChatLanguageModel("text"),
      messages: [
        { role: "system", content: "You are a helpful assistant." },
        { role: "user", content: "Who are you?" },
      ],
    });

    expect(result.text).toBe("I am a helpful assistant.");

    // Verify system prompt is passed via initialPrompts (Prompt API spec)
    const createCall = (globalThis.LanguageModel as any).create.mock
      .calls[0][0];
    expect(createCall.initialPrompts).toEqual([
      { role: "system", content: "You are a helpful assistant." },
    ]);

    // Verify messages passed to prompt() don't include system prompt
    const [messagesArg, optionsArg] = mockPrompt.mock.calls[0];
    expect(optionsArg).toEqual({});
    expect(messagesArg).toHaveLength(1);
    expect(messagesArg[0].role).toBe("user");
    expect(messagesArg[0].content).toEqual([
      { type: "text", value: "Who are you?" },
    ]);
  });

  it("should handle conversation history", async () => {
    mockPrompt.mockResolvedValue("I can help you with that!");

    const result = await generateText({
      model: new BrowserAIChatLanguageModel("text"),
      messages: [
        { role: "user", content: "Can you help me?" },
        { role: "assistant", content: "Of course! What do you need?" },
        { role: "user", content: "I need assistance with coding." },
      ],
    });

    expect(result.text).toBe("I can help you with that!");
    expect(mockPrompt).toHaveBeenCalledWith(
      [
        {
          role: "user",
          content: [{ type: "text", value: "Can you help me?" }],
        },
        {
          role: "assistant",
          content: "Of course! What do you need?",
        },
        {
          role: "user",
          content: [{ type: "text", value: "I need assistance with coding." }],
        },
      ],
      {},
    );
  });

  it("should stream text successfully", async () => {
    const mockStream = new ReadableStream({
      start(controller) {
        controller.enqueue("Hello");
        controller.enqueue(", ");
        controller.enqueue("world!");
        controller.close();
      },
    });

    mockPromptStreaming.mockReturnValue(mockStream);

    const result = await streamText({
      model: new BrowserAIChatLanguageModel("text"),
      prompt: "Say hello",
    });

    let text = "";
    for await (const chunk of result.textStream) {
      text += chunk;
    }

    expect(text).toBe("Hello, world!");
    expect(mockPromptStreaming).toHaveBeenCalledWith(
      [
        {
          role: "user",
          content: [{ type: "text", value: "Say hello" }],
        },
      ],
      {
        signal: undefined,
      },
    );
  });

  it("should handle JSON response format", async () => {
    const jsonResponse = JSON.stringify({ name: "John", age: 30 });
    mockPrompt.mockResolvedValue(jsonResponse);

    const schema = z.object({
      name: z.string(),
      age: z.number(),
    });

    const { object } = await generateObject({
      model: new BrowserAIChatLanguageModel("text"),
      schema,
      prompt: "Create a person",
    });

    expect(object).toEqual({ name: "John", age: 30 });
    expect(mockPrompt).toHaveBeenCalledWith(
      [
        {
          role: "user",
          content: [{ type: "text", value: "Create a person" }],
        },
      ],
      {
        responseConstraint: {
          $schema: "http://json-schema.org/draft-07/schema#",
          additionalProperties: false,
          properties: {
            age: { type: "number" },
            name: { type: "string" },
          },
          required: ["name", "age"],
          type: "object",
        },
      },
    );
  });

  it("should handle object generation mode", async () => {
    const jsonResponse = JSON.stringify({ users: ["Alice", "Bob"] });
    mockPrompt.mockResolvedValue(jsonResponse);

    const schema = z.object({
      users: z.array(z.string()),
    });

    const { object } = await generateObject({
      model: new BrowserAIChatLanguageModel("text"),
      schema,
      prompt: "List some users",
    });

    expect(object).toEqual({ users: ["Alice", "Bob"] });
    expect(mockPrompt).toHaveBeenCalledWith(
      [
        {
          role: "user",
          content: [{ type: "text", value: "List some users" }],
        },
      ],
      {
        responseConstraint: {
          $schema: "http://json-schema.org/draft-07/schema#",
          additionalProperties: false,
          properties: {
            users: {
              items: { type: "string" },
              type: "array",
            },
          },
          required: ["users"],
          type: "object",
        },
      },
    );
  });

  it("should handle complex JSON schemas", async () => {
    const jsonResponse = JSON.stringify({
      users: [
        { id: 1, name: "Alice", active: true },
        { id: 2, name: "Bob", active: false },
      ],
      total: 2,
    });

    mockPrompt.mockResolvedValue(jsonResponse);

    const schema = z.object({
      users: z.array(
        z.object({
          id: z.number(),
          name: z.string(),
          active: z.boolean(),
        }),
      ),
      total: z.number(),
    });

    const { object } = await generateObject({
      model: new BrowserAIChatLanguageModel("text"),
      schema,
      prompt: "Create a user list",
    });

    expect(object).toEqual({
      users: [
        { id: 1, name: "Alice", active: true },
        { id: 2, name: "Bob", active: false },
      ],
      total: 2,
    });
  });

  it("should handle empty content arrays", async () => {
    mockPrompt.mockResolvedValue("Response");

    const result = await generateText({
      model: new BrowserAIChatLanguageModel("text"),
      messages: [
        {
          role: "user",
          content: [],
        },
      ],
    });

    expect(result.text).toBe("Response");
    expect(mockPrompt).toHaveBeenCalledWith(
      [
        {
          role: "user",
          content: [],
        },
      ],
      {},
    );
  });

  describe("multimodal support", () => {
    beforeEach(() => {
      // Mock LanguageModel.create to capture the options passed to it
      LanguageModel.create = vi.fn().mockResolvedValue(mockSession);
    });

    it("should handle image files in messages", async () => {
      mockPrompt.mockResolvedValue("I can see an image.");

      const result = await generateText({
        model: new BrowserAIChatLanguageModel("text"),
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "What's in this image?" },
              {
                type: "file",
                mediaType: "image/png",
                data: "SGVsbG8gV29ybGQ=", // "Hello World" in base64
              },
            ],
          },
        ],
      });

      expect(result.text).toBe("I can see an image.");

      // Verify that the session was created with expected inputs for image
      expect(LanguageModel.create).toHaveBeenCalledWith(
        expect.objectContaining<Partial<BrowserAIChatSettings>>({
          expectedInputs: [{ type: "image" }],
        }),
      );
    });

    it("should handle audio files in messages", async () => {
      mockPrompt.mockResolvedValue("I can hear the audio.");

      const result = await generateText({
        model: new BrowserAIChatLanguageModel("text"),
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "What's in this audio?" },
              {
                type: "file",
                mediaType: "audio/wav",
                data: new Uint8Array([82, 73, 70, 70]), // "RIFF" header
              },
            ],
          },
        ],
      });

      expect(result.text).toBe("I can hear the audio.");

      // Verify that the session was created with expected inputs for audio
      expect(LanguageModel.create).toHaveBeenCalledWith(
        expect.objectContaining<Partial<BrowserAIChatSettings>>({
          expectedInputs: [{ type: "audio" }],
        }),
      );
    });

    it("should handle both image and audio content", async () => {
      mockPrompt.mockResolvedValue("I can see and hear the content.");

      const result = await generateText({
        model: new BrowserAIChatLanguageModel("text"),
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "Analyze this:" },
              {
                type: "file",
                mediaType: "image/jpeg",
                data: "SGVsbG8=", // "Hello" in base64
              },
              { type: "text", text: "And this:" },
              {
                type: "file",
                mediaType: "audio/mp3",
                data: new Uint8Array([1, 2, 3]),
              },
            ],
          },
        ],
      });

      expect(result.text).toBe("I can see and hear the content.");

      // Verify that the session was created with expected inputs for both image and audio
      expect(LanguageModel.create).toHaveBeenCalledWith(
        expect.objectContaining<Partial<BrowserAIChatSettings>>({
          expectedInputs: expect.arrayContaining([
            { type: "image" },
            { type: "audio" },
          ]),
        }),
      );
    });

    it("should handle URL-based image data", async () => {
      mockPrompt.mockResolvedValue("I can see the image from the URL.");

      const result = await generateText({
        model: new BrowserAIChatLanguageModel("text"),
        messages: [
          {
            role: "user",
            content: [
              {
                type: "file",
                mediaType: "image/png",
                data: new URL("https://example.com/image.png"),
              },
            ],
          },
        ],
      });

      expect(result.text).toBe("I can see the image from the URL.");

      // Verify that the session was created with expected inputs for image
      expect(LanguageModel.create).toHaveBeenCalledWith(
        expect.objectContaining<Partial<BrowserAIChatSettings>>({
          expectedInputs: [{ type: "image" }],
        }),
      );
    });
  });

  describe("tool support", () => {
    it("should return tool call parts when the model requests a tool", async () => {
      mockPrompt.mockResolvedValue(`Checking the weather.
\`\`\`tool_call
{"name": "getWeather", "arguments": {"location": "Seattle"}}
\`\`\`
Running the tool now.`);

      const model = new BrowserAIChatLanguageModel("text");

      const response = await model.doGenerate({
        prompt: [
          {
            role: "user",
            content: [
              { type: "text", text: "What is the weather in Seattle?" },
            ],
          },
        ],
        tools: [
          {
            type: "function",
            name: "getWeather",
            description: "Get the weather in a location.",
            inputSchema: {
              type: "object",
              properties: {
                location: { type: "string" },
              },
              required: ["location"],
            },
          },
        ],
      });

      expect(response.finishReason).toMatchObject({
        raw: "tool-calls",
        unified: "tool-calls",
      });
      expect(response.content).toEqual([
        { type: "text", text: "Checking the weather.\nRunning the tool now." },
        {
          type: "tool-call",
          toolCallId: expect.any(String),
          toolName: "getWeather",
          input: '{"location":"Seattle"}',
        },
      ]);

      // Verify tool instructions are passed via initialPrompts (Prompt API spec)
      const createCall = (globalThis.LanguageModel as any).create.mock
        .calls[0][0];
      expect(createCall.initialPrompts[0].role).toBe("system");
      expect(createCall.initialPrompts[0].content).toContain("getWeather");
      expect(createCall.initialPrompts[0].content).toContain("```tool_call");
      expect(createCall.initialPrompts[0].content).toContain("Available Tools");

      // Verify messages passed to prompt() are the raw user messages
      const promptCallArgs = mockPrompt.mock.calls[0][0] as any[];
      const firstUserMessage = promptCallArgs[0];
      expect(firstUserMessage.role).toBe("user");
      expect(firstUserMessage.content[0].value).toBe(
        "What is the weather in Seattle?",
      );
    });

    it("should emit only the first tool call when parallel execution is disabled", async () => {
      mockPrompt.mockResolvedValue(
        `\`\`\`tool_call
{"name": "getWeather", "arguments": {"location": "Seattle"}}
{"name": "getNews", "arguments": {"topic": "Seattle"}}
\`\`\`
I'll follow up once I have the results.`,
      );

      const model = new BrowserAIChatLanguageModel("text");

      const response = await model.doGenerate({
        prompt: [
          {
            role: "user",
            content: [{ type: "text", text: "What's happening in Seattle?" }],
          },
        ],
        tools: [
          {
            type: "function",
            name: "getWeather",
            description: "Get the weather in a location.",
            inputSchema: {
              type: "object",
              properties: {
                location: { type: "string" },
              },
              required: ["location"],
            },
          },
          {
            type: "function",
            name: "getNews",
            description: "Get the latest news.",
            inputSchema: {
              type: "object",
              properties: {
                topic: { type: "string" },
              },
              required: ["topic"],
            },
          },
        ],
      });

      const toolCalls = response.content.filter(
        (part) => part.type === "tool-call",
      );

      expect(toolCalls).toHaveLength(1);
      expect(toolCalls[0]).toMatchObject({
        toolName: "getWeather",
        input: '{"location":"Seattle"}',
      });
    });

    it("should emit tool call events during streaming", async () => {
      const streamingResponse = `Checking the weather.
\`\`\`tool_call
{"name": "getWeather", "arguments": {"location": "Seattle"}}
\`\`\`
Running the tool now.`;

      mockPromptStreaming.mockReturnValue(
        new ReadableStream<string>({
          start(controller) {
            controller.enqueue(streamingResponse);
            controller.close();
          },
        }),
      );

      const model = new BrowserAIChatLanguageModel("text");

      const { stream } = await model.doStream({
        prompt: [
          {
            role: "user",
            content: [
              { type: "text", text: "What is the weather in Seattle?" },
            ],
          },
        ],
        tools: [
          {
            type: "function",
            name: "getWeather",
            description: "Get the weather in a location.",
            inputSchema: {
              type: "object",
              properties: {
                location: { type: "string" },
              },
              required: ["location"],
            },
          },
        ],
      });

      const events: LanguageModelV3StreamPart[] = [];
      const reader = stream.getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        events.push(value);
      }

      expect(events[0]).toMatchObject({ type: "stream-start" });
      expect(events).toContainEqual({ type: "text-start", id: "text-0" });

      const textDeltas = events
        .filter(
          (
            event,
          ): event is Extract<
            LanguageModelV3StreamPart,
            { type: "text-delta" }
          > => event.type === "text-delta",
        )
        .map((event) => event.delta.trim());
      expect(textDeltas).toEqual([
        "Checking the weather.",
        "Running the tool now.",
      ]);

      const toolEvent = events.find(
        (
          event,
        ): event is Extract<LanguageModelV3StreamPart, { type: "tool-call" }> =>
          event.type === "tool-call",
      );

      expect(toolEvent).toMatchObject({
        toolName: "getWeather",
        input: '{"location":"Seattle"}',
        providerExecuted: false,
      });

      const finishEvent = events.find(
        (
          event,
        ): event is Extract<LanguageModelV3StreamPart, { type: "finish" }> =>
          event.type === "finish",
      );

      expect(finishEvent).toMatchObject({
        finishReason: { raw: "tool-calls", unified: "tool-calls" },
      });

      // Verify tool instructions are passed via initialPrompts (Prompt API spec)
      const createCall = (globalThis.LanguageModel as any).create.mock
        .calls[0][0];
      expect(createCall.initialPrompts[0].role).toBe("system");
      expect(createCall.initialPrompts[0].content).toContain("getWeather");
      expect(createCall.initialPrompts[0].content).toContain("```tool_call");
      expect(createCall.initialPrompts[0].content).toContain("Available Tools");

      // Verify messages passed to promptStreaming are the raw user messages
      const promptCallArgs = mockPromptStreaming.mock.calls[0][0];
      expect(promptCallArgs[0].role).toBe("user");
      expect(promptCallArgs[0].content[0].value).toBe(
        "What is the weather in Seattle?",
      );
    });

    it("should emit only the first streaming tool call when parallel execution is disabled", async () => {
      const streamingResponse = `\`\`\`tool_call
{"name": "getWeather", "arguments": {"location": "Seattle"}}
{"name": "getNews", "arguments": {"topic": "Seattle"}}
\`\`\`
Running the tool now.`;

      mockPromptStreaming.mockReturnValue(
        new ReadableStream<string>({
          start(controller) {
            controller.enqueue(streamingResponse);
            controller.close();
          },
        }),
      );

      const model = new BrowserAIChatLanguageModel("text");

      const { stream } = await model.doStream({
        prompt: [
          {
            role: "user",
            content: [{ type: "text", text: "What's happening in Seattle?" }],
          },
        ],
        tools: [
          {
            type: "function",
            name: "getWeather",
            description: "Get the weather in a location.",
            inputSchema: {
              type: "object",
              properties: {
                location: { type: "string" },
              },
              required: ["location"],
            },
          },
          {
            type: "function",
            name: "getNews",
            description: "Get the latest news.",
            inputSchema: {
              type: "object",
              properties: {
                topic: { type: "string" },
              },
              required: ["topic"],
            },
          },
        ],
      });

      const events: LanguageModelV3StreamPart[] = [];
      const reader = stream.getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        events.push(value);
      }

      const toolEvents = events.filter((event) => event.type === "tool-call");

      expect(toolEvents).toHaveLength(1);
      expect(toolEvents[0]).toMatchObject({
        toolName: "getWeather",
        input: '{"location":"Seattle"}',
      });

      const finishEvent = events.find((event) => event.type === "finish");
      expect(finishEvent).toMatchObject({
        finishReason: { raw: "tool-calls", unified: "tool-calls" },
      });
    });

    it("should use consistent tool call ID across all streaming events", async () => {
      const streamingResponse = `\`\`\`tool_call
{"name": "getWeather", "arguments": {"location": "Seattle"}}
\`\`\``;

      mockPromptStreaming.mockReturnValue(
        new ReadableStream<string>({
          start(controller) {
            controller.enqueue(streamingResponse);
            controller.close();
          },
        }),
      );

      const model = new BrowserAIChatLanguageModel("text");

      const { stream } = await model.doStream({
        prompt: [
          {
            role: "user",
            content: [
              { type: "text", text: "What is the weather in Seattle?" },
            ],
          },
        ],
        tools: [
          {
            type: "function",
            name: "getWeather",
            description: "Get the weather in a location.",
            inputSchema: {
              type: "object",
              properties: {
                location: { type: "string" },
              },
              required: ["location"],
            },
          },
        ],
      });

      const events: LanguageModelV3StreamPart[] = [];
      const reader = stream.getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        events.push(value);
      }

      // Extract tool-related events
      const toolInputStartEvent = events.find(
        (
          event,
        ): event is Extract<
          LanguageModelV3StreamPart,
          { type: "tool-input-start" }
        > => event.type === "tool-input-start",
      );
      const toolInputDeltaEvents = events.filter(
        (
          event,
        ): event is Extract<
          LanguageModelV3StreamPart,
          { type: "tool-input-delta" }
        > => event.type === "tool-input-delta",
      );
      const toolInputEndEvent = events.find(
        (
          event,
        ): event is Extract<
          LanguageModelV3StreamPart,
          { type: "tool-input-end" }
        > => event.type === "tool-input-end",
      );
      const toolCallEvent = events.find(
        (
          event,
        ): event is Extract<LanguageModelV3StreamPart, { type: "tool-call" }> =>
          event.type === "tool-call",
      );

      // Verify all events exist
      expect(toolInputStartEvent).toBeDefined();
      expect(toolInputDeltaEvents.length).toBeGreaterThan(0);
      expect(toolInputEndEvent).toBeDefined();
      expect(toolCallEvent).toBeDefined();

      // CRITICAL: All events must use the SAME tool call ID
      const toolCallId = toolInputStartEvent!.id;
      expect(toolCallId).toBeTruthy();

      // Verify tool-input-delta events all use the same ID
      for (const deltaEvent of toolInputDeltaEvents) {
        expect(deltaEvent.id).toBe(toolCallId);
      }

      // Verify tool-input-end uses the same ID
      expect(toolInputEndEvent!.id).toBe(toolCallId);

      // Verify tool-call uses the same ID
      expect(toolCallEvent!.toolCallId).toBe(toolCallId);

      // Additional verification: ensure we don't have multiple different tool call IDs
      const allToolCallIds = new Set([
        toolInputStartEvent!.id,
        ...toolInputDeltaEvents.map((e) => e.id),
        toolInputEndEvent!.id,
        toolCallEvent!.toolCallId,
      ]);

      expect(allToolCallIds.size).toBe(1); // All IDs should be identical
    });
  });

  describe("createSessionWithProgress", () => {
    let mockEventTarget: {
      addEventListener: ReturnType<typeof vi.fn>;
      removeEventListener: ReturnType<typeof vi.fn>;
      dispatchEvent: ReturnType<typeof vi.fn>;
      ondownloadprogress: null;
    };

    beforeEach(() => {
      // Create a mock CreateMonitor that matches the DOM API
      mockEventTarget = {
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn(),
        ondownloadprogress: null,
      };

      // Mock LanguageModel.create to capture monitor option and simulate its usage
      LanguageModel.create = vi.fn((options: LanguageModelCreateOptions) => {
        // If a monitor option is provided, call it to set up event listeners
        if (options.monitor) {
          options.monitor(mockEventTarget as CreateMonitor);
        }
        return Promise.resolve(mockSession);
      });
    });

    it("should create a session without progress callback", async () => {
      const model = new BrowserAIChatLanguageModel("text");
      const session = await model.createSessionWithProgress();

      expect(session).toBe(mockSession);
      expect(LanguageModel.create).toHaveBeenCalledWith(
        expect.not.objectContaining({
          monitor: expect.any(Function),
        }),
      );
    });

    it("should create a session with progress callback and forward progress events", async () => {
      const model = new BrowserAIChatLanguageModel("text");
      const progressCallback = vi.fn();

      // Mock LanguageModel.create to simulate progress events
      LanguageModel.create = vi.fn((options: LanguageModelCreateOptions) => {
        if (options.monitor) {
          options.monitor(mockEventTarget as CreateMonitor);

          // Simulate the addEventListener call and trigger progress events
          const addEventListenerCall =
            mockEventTarget.addEventListener.mock.calls.find(
              (call) => call[0] === "downloadprogress",
            );

          if (addEventListenerCall) {
            const progressHandler = addEventListenerCall[1];

            // Simulate progress events
            setTimeout(() => {
              progressHandler({ loaded: 0.0 });
              progressHandler({ loaded: 0.5 });
              progressHandler({ loaded: 1.0 });
            }, 0);
          }
        }
        return Promise.resolve(mockSession);
      });

      const session = await model.createSessionWithProgress(progressCallback);

      expect(session).toBe(mockSession);
      expect(LanguageModel.create).toHaveBeenCalledWith(
        expect.objectContaining({
          monitor: expect.any(Function),
        }),
      );
      expect(mockEventTarget.addEventListener).toHaveBeenCalledWith(
        "downloadprogress",
        expect.any(Function),
      );

      // Wait for the setTimeout to complete
      await new Promise((resolve) => setTimeout(resolve, 10));

      expect(progressCallback).toHaveBeenCalledTimes(3);
      expect(progressCallback).toHaveBeenNthCalledWith(1, 0.0);
      expect(progressCallback).toHaveBeenNthCalledWith(2, 0.5);
      expect(progressCallback).toHaveBeenNthCalledWith(3, 1.0);
    });

    it("should reuse existing session on subsequent calls", async () => {
      const model = new BrowserAIChatLanguageModel("text");

      // First call should create a new session
      const session1 = await model.createSessionWithProgress();
      expect(session1).toBe(mockSession);
      expect(LanguageModel.create).toHaveBeenCalledTimes(1);

      // Second call should reuse the existing session
      const session2 = await model.createSessionWithProgress();
      expect(session2).toBe(mockSession);
      expect(session1).toBe(session2);
      expect(LanguageModel.create).toHaveBeenCalledTimes(1);
    });

    it("should throw LoadSettingError when LanguageModel is unavailable", async () => {
      vi.stubGlobal("LanguageModel", undefined);
      const model = new BrowserAIChatLanguageModel("text");

      await expect(model.createSessionWithProgress()).rejects.toThrow(
        LoadSettingError,
      );
    });
  });
});
