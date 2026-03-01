import { describe, it, expect } from "vitest";
import { convertToTransformersMessages } from "../src/utils/convert-to-transformers-message";
import {
  UnsupportedFunctionalityError,
  type LanguageModelV3Prompt,
} from "@ai-sdk/provider";

describe("convertToTransformersMessages", () => {
  it("converts simple text user message", () => {
    const prompt: LanguageModelV3Prompt = [
      { role: "user", content: [{ type: "text", text: "Hello" }] },
    ];

    const result = convertToTransformersMessages(prompt);
    expect(result).toEqual([{ role: "user", content: "Hello" }]);
  });

  it("converts assistant text message", () => {
    const prompt: LanguageModelV3Prompt = [
      { role: "assistant", content: [{ type: "text", text: "Hi" }] },
    ];

    const result = convertToTransformersMessages(prompt);
    expect(result).toEqual([{ role: "assistant", content: "Hi" }]);
  });

  it("keeps system content as-is", () => {
    const prompt: LanguageModelV3Prompt = [
      { role: "system", content: "You are helpful." },
    ];

    const result = convertToTransformersMessages(prompt);
    expect(result).toEqual([{ role: "system", content: "You are helpful." }]);
  });

  it("throws for non-vision file input in user message", () => {
    const prompt: LanguageModelV3Prompt = [
      {
        role: "user",
        content: [
          { type: "text", text: "See this" },
          { type: "file", mediaType: "image/png", data: "AAA" },
        ],
      },
    ];
    expect(() => convertToTransformersMessages(prompt)).toThrow(
      UnsupportedFunctionalityError,
    );
  });

  it("converts image content when isVisionModel=true", () => {
    const base64 = "SGVsbG8="; // Hello
    const prompt: LanguageModelV3Prompt = [
      {
        role: "user",
        content: [
          { type: "text", text: "What is in this image?" },
          { type: "file", mediaType: "image/png", data: base64 },
          { type: "text", text: "Thanks" },
        ],
      },
    ];

    const result = convertToTransformersMessages(prompt, true);
    expect(result).toHaveLength(1);
    expect(result[0].role).toBe("user");
    expect(Array.isArray(result[0].content)).toBe(true);
    const parts = result[0].content as any[];
    expect(parts).toEqual([
      { type: "text", text: "What is in this image?" },
      {
        type: "image",
        image: `data:image/png;base64,${base64}`,
      },
      { type: "text", text: "Thanks" },
    ]);
  });

  it("converts tool role to native HF format with role 'tool'", () => {
    const prompt = [
      {
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "call_123",
            toolName: "get_weather",
            output: { type: "text", value: "72°F and sunny" },
          },
        ],
      },
    ] as any;
    const result = convertToTransformersMessages(prompt);
    expect(result).toHaveLength(1);
    expect(result[0].role).toBe("tool");
    expect(result[0]).toHaveProperty("tool_call_id", "call_123");
    expect(result[0]).toHaveProperty("name", "get_weather");
    expect(result[0]).toHaveProperty("content", "72°F and sunny");
  });

  it("converts assistant tool-call content to native HF format", () => {
    const prompt: LanguageModelV3Prompt = [
      {
        role: "assistant",
        content: [
          {
            type: "tool-call",
            toolCallId: "call_456",
            toolName: "calculate",
            input: { x: 5, y: 10 },
          } as any,
        ],
      },
    ];
    const result = convertToTransformersMessages(prompt);
    expect(result).toHaveLength(1);
    expect(result[0].role).toBe("assistant");
    expect(result[0].content).toBe(null);
    expect(result[0]).toHaveProperty("tool_calls");
    expect(result[0].tool_calls).toHaveLength(1);
    expect(result[0].tool_calls![0]).toEqual({
      id: "call_456",
      type: "function",
      function: {
        name: "calculate",
        arguments: '{"x":5,"y":10}',
      },
    });
  });

  it("converts assistant with both text and tool-call", () => {
    const prompt: LanguageModelV3Prompt = [
      {
        role: "assistant",
        content: [
          { type: "text", text: "Let me calculate that for you." },
          {
            type: "tool-call",
            toolCallId: "call_789",
            toolName: "add",
            input: { a: 2, b: 3 },
          } as any,
        ],
      },
    ];
    const result = convertToTransformersMessages(prompt);
    expect(result).toHaveLength(1);
    expect(result[0].role).toBe("assistant");
    expect(result[0].content).toBe("Let me calculate that for you.");
    expect(result[0]).toHaveProperty("tool_calls");
    expect(result[0].tool_calls).toHaveLength(1);
    expect(result[0].tool_calls![0].function.name).toBe("add");
  });

  it("converts multiple tool results into separate messages", () => {
    const prompt = [
      {
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "call_1",
            toolName: "get_weather",
            output: { type: "text", value: "Sunny" },
          },
          {
            type: "tool-result",
            toolCallId: "call_2",
            toolName: "get_time",
            output: { type: "text", value: "12:00 PM" },
          },
        ],
      },
    ] as any;
    const result = convertToTransformersMessages(prompt);
    expect(result).toHaveLength(2);
    expect(result[0].role).toBe("tool");
    expect(result[0]).toHaveProperty("tool_call_id", "call_1");
    expect(result[0]).toHaveProperty("name", "get_weather");
    expect(result[1].role).toBe("tool");
    expect(result[1]).toHaveProperty("tool_call_id", "call_2");
    expect(result[1]).toHaveProperty("name", "get_time");
  });
});
