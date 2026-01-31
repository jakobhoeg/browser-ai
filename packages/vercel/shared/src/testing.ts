/**
 * Test helpers for @browser-ai packages
 *
 * Import from "@browser-ai/shared/testing" in test files
 */

// ============================================================================
// WebGPU / Navigator Mocks
// ============================================================================

let originalGpu: unknown;

/**
 * Mocks the navigator.gpu property for testing WebGPU-dependent code
 *
 * @param value - The value to set for navigator.gpu (default: empty object to simulate WebGPU support)
 *
 * @example
 * ```typescript
 * beforeEach(() => {
 *   mockWebGPU(); // Simulate WebGPU support
 * });
 *
 * it("should work without WebGPU", () => {
 *   mockWebGPU(undefined); // Simulate no WebGPU
 *   // ... test code
 * });
 * ```
 */
export function mockWebGPU(value: object | undefined = {}): void {
  if (typeof originalGpu === "undefined") {
    originalGpu = (globalThis.navigator as any)?.gpu;
  }

  Object.defineProperty(globalThis.navigator, "gpu", {
    value,
    configurable: true,
    writable: true,
  });
}

/**
 * Restores the original navigator.gpu value
 */
export function restoreWebGPU(): void {
  Object.defineProperty(globalThis.navigator, "gpu", {
    value: originalGpu,
    configurable: true,
    writable: true,
  });
}

// ============================================================================
// Stream Helpers
// ============================================================================

/**
 * Reads all values from a ReadableStream into an array
 *
 * @param stream - The stream to read
 * @returns Array of all values from the stream
 *
 * @example
 * ```typescript
 * const result = await model.doStream({ prompt });
 * const parts = await readStream(result.stream);
 * expect(parts).toContainEqual({ type: "text-delta", textDelta: "Hello" });
 * ```
 */
export async function readStream<T>(stream: ReadableStream<T>): Promise<T[]> {
  const parts: T[] = [];
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

  return parts;
}

/**
 * Creates an async generator from an array of items
 *
 * @param items - Items to yield from the generator
 * @returns An async generator that yields each item
 *
 * @example
 * ```typescript
 * const mockResponse = createAsyncGenerator([
 *   { choices: [{ delta: { content: "Hello" } }] },
 *   { choices: [{ delta: { content: " world" } }] },
 * ]);
 * mockChatCompletionsCreate.mockReturnValue(mockResponse);
 * ```
 */
export async function* createAsyncGenerator<T>(
  items: T[],
): AsyncGenerator<T, void, unknown> {
  for (const item of items) {
    yield item;
  }
}

// ============================================================================
// Tool Fixtures
// ============================================================================

/**
 * Common weather tool fixture for testing tool calling
 */
export const weatherTool = {
  type: "function" as const,
  name: "get_weather",
  description: "Get the weather for a city",
  inputSchema: {
    type: "object" as const,
    properties: {
      city: { type: "string" as const },
    },
    required: ["city"],
  },
};

/**
 * Common search tool fixture for testing tool calling
 */
export const searchTool = {
  type: "function" as const,
  name: "search",
  description: "Search for information",
  inputSchema: {
    type: "object" as const,
    properties: {
      query: { type: "string" as const },
    },
    required: ["query"],
  },
};

/**
 * Creates a tool definition for testing
 *
 * @param name - Tool name
 * @param description - Tool description
 * @param properties - Input schema properties
 * @param required - Required properties
 *
 * @example
 * ```typescript
 * const myTool = createTool("my_tool", "Does something", {
 *   input: { type: "string" }
 * }, ["input"]);
 * ```
 */
export function createTool(
  name: string,
  description: string,
  properties: Record<string, { type: string }>,
  required: string[] = [],
) {
  return {
    type: "function" as const,
    name,
    description,
    inputSchema: {
      type: "object" as const,
      properties,
      required,
    },
  };
}

// ============================================================================
// Prompt Helpers
// ============================================================================

/**
 * Creates a simple user prompt for testing
 *
 * @param text - The user message text
 *
 * @example
 * ```typescript
 * const result = await model.doGenerate({
 *   prompt: userPrompt("Hello"),
 * });
 * ```
 */
export function userPrompt(text: string) {
  return [
    {
      role: "user" as const,
      content: [{ type: "text" as const, text }],
    },
  ];
}

/**
 * Creates a prompt with system and user messages
 *
 * @param system - The system message
 * @param user - The user message
 *
 * @example
 * ```typescript
 * const result = await model.doGenerate({
 *   prompt: systemUserPrompt("You are helpful", "Hello"),
 * });
 * ```
 */
export function systemUserPrompt(system: string, user: string) {
  return [
    {
      role: "system" as const,
      content: system,
    },
    {
      role: "user" as const,
      content: [{ type: "text" as const, text: user }],
    },
  ];
}

// ============================================================================
// Mock Response Factories
// ============================================================================

/**
 * Creates a mock embedding response
 *
 * @param embeddings - Array of embedding vectors
 * @param model - Model name (default: "test-model")
 *
 * @example
 * ```typescript
 * mockEmbeddingsCreate.mockResolvedValue(
 *   createEmbeddingResponse([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
 * );
 * ```
 */
export function createEmbeddingResponse(
  embeddings: number[][],
  model: string = "test-model",
) {
  return {
    data: embeddings.map((embedding, index) => ({
      index,
      embedding,
      object: "embedding" as const,
    })),
    model,
    usage: {
      prompt_tokens: embeddings.length * 5,
      total_tokens: embeddings.length * 5,
      extra: {},
    },
  };
}

/**
 * Creates a mock chat completion response
 *
 * @param content - The response content
 * @param finishReason - The finish reason (default: "stop")
 *
 * @example
 * ```typescript
 * mockChatCompletionsCreate.mockResolvedValue(
 *   createChatResponse("Hello, world!")
 * );
 * ```
 */
export function createChatResponse(
  content: string,
  finishReason: string = "stop",
) {
  return {
    choices: [
      {
        message: { content },
        finish_reason: finishReason,
      },
    ],
    usage: {
      prompt_tokens: 10,
      completion_tokens: content.split(" ").length,
      total_tokens: 10 + content.split(" ").length,
    },
  };
}

/**
 * Creates a mock streaming chat chunk
 *
 * @param content - The content delta
 * @param finishReason - Optional finish reason
 *
 * @example
 * ```typescript
 * const stream = createAsyncGenerator([
 *   createStreamChunk("Hello"),
 *   createStreamChunk(", world!"),
 *   createStreamChunk("", "stop"),
 * ]);
 * ```
 */
export function createStreamChunk(content: string, finishReason?: string) {
  return {
    choices: [
      {
        delta: { content: content || undefined },
        finish_reason: finishReason ?? null,
      },
    ],
    ...(finishReason && {
      usage: {
        prompt_tokens: 10,
        completion_tokens: 5,
        total_tokens: 15,
      },
    }),
  };
}

/**
 * Creates a tool call response string in markdown fence format
 *
 * @param name - Tool name
 * @param args - Tool arguments
 *
 * @example
 * ```typescript
 * const response = createToolCallResponse("get_weather", { city: "London" });
 * // Returns: "```tool_call\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"London\"}}\n```"
 * ```
 */
export function createToolCallResponse(
  name: string,
  args: Record<string, unknown>,
): string {
  return `\`\`\`tool_call
{"name": "${name}", "arguments": ${JSON.stringify(args)}}
\`\`\``;
}
