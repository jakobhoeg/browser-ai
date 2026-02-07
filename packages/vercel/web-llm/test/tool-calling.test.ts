import { describe, it, expect } from "vitest";
import {
  parseJsonFunctionCalls,
  buildJsonToolSystemPrompt,
  type ToolDefinition,
} from "@browser-ai/shared";
import { extractSystemPrompt } from "../src/utils/prompt-utils";

describe("parseJsonFunctionCalls", () => {
  it("returns empty array for text without tool calls", () => {
    const result = parseJsonFunctionCalls("This is just plain text");
    expect(result.toolCalls).toEqual([]);
    expect(result.textContent).toBe("This is just plain text");
  });

  it("parses single tool call", () => {
    const response = `\`\`\`tool_call
{"name": "get_weather", "arguments": {"city": "SF"}}
\`\`\``;

    const result = parseJsonFunctionCalls(response);
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0].toolName).toBe("get_weather");
    expect(result.toolCalls[0].args).toEqual({ city: "SF" });
    expect(result.toolCalls[0].toolCallId).toMatch(/^call_/);
  });

  it("parses tool_call and tool-call fence variants", () => {
    const underscore = parseJsonFunctionCalls(
      '```tool_call\n{"name": "test"}\n```',
    );
    const hyphen = parseJsonFunctionCalls(
      '```tool-call\n{"name": "test"}\n```',
    );
    const noSpace = parseJsonFunctionCalls(
      '```toolcall\n{"name": "test"}\n```',
    );

    expect(underscore.toolCalls).toHaveLength(1);
    expect(hyphen.toolCalls).toHaveLength(1);
    expect(noSpace.toolCalls).toHaveLength(1);
  });

  it("preserves custom ID or generates one", () => {
    const withId = parseJsonFunctionCalls(
      '```tool_call\n{"id": "custom_123", "name": "test"}\n```',
    );
    const withoutId = parseJsonFunctionCalls(
      '```tool_call\n{"name": "test"}\n```',
    );

    expect(withId.toolCalls[0].toolCallId).toBe("custom_123");
    expect(withoutId.toolCalls[0].toolCallId).toMatch(/^call_\d+_[a-z0-9]{7}$/);
  });

  it("handles missing or empty arguments", () => {
    const noArgs = parseJsonFunctionCalls(
      '```tool_call\n{"name": "test"}\n```',
    );
    const emptyArgs = parseJsonFunctionCalls(
      '```tool_call\n{"name": "test", "arguments": {}}\n```',
    );

    expect(noArgs.toolCalls[0].args).toEqual({});
    expect(emptyArgs.toolCalls[0].args).toEqual({});
  });

  it("parses array of tool calls", () => {
    const response = `\`\`\`tool_call
[
  {"name": "tool1", "arguments": {"a": 1}},
  {"name": "tool2", "arguments": {"b": 2}}
]
\`\`\``;

    const result = parseJsonFunctionCalls(response);
    expect(result.toolCalls).toHaveLength(2);
    expect(result.toolCalls[0].toolName).toBe("tool1");
    expect(result.toolCalls[1].toolName).toBe("tool2");
  });

  it("parses newline-separated tool calls", () => {
    const response = `\`\`\`tool_call
{"name": "first", "arguments": {"x": 1}}
{"name": "second", "arguments": {"y": 2}}
\`\`\``;

    const result = parseJsonFunctionCalls(response);
    expect(result.toolCalls).toHaveLength(2);
  });

  it("extracts text content and removes fences", () => {
    const response = `Let me help.
\`\`\`tool_call
{"name": "help", "arguments": {}}
\`\`\`
Done!`;

    const result = parseJsonFunctionCalls(response);
    expect(result.textContent).toContain("Let me help.");
    expect(result.textContent).toContain("Done!");
    expect(result.textContent).not.toContain("```");
  });

  it("handles invalid JSON gracefully", () => {
    const invalid = parseJsonFunctionCalls("```tool_call\n{invalid}\n```");
    const noName = parseJsonFunctionCalls(
      '```tool_call\n{"arguments": {}}\n```',
    );

    expect(invalid.toolCalls).toEqual([]);
    expect(noName.toolCalls).toEqual([]);
  });

  it("parses complex nested arguments", () => {
    const response = `\`\`\`tool_call
{
  "name": "test",
  "arguments": {
    "nested": {"level": "deep"},
    "array": [1, "two", true],
    "null": null
  }
}
\`\`\``;

    const result = parseJsonFunctionCalls(response);
    expect(result.toolCalls[0].args).toEqual({
      nested: { level: "deep" },
      array: [1, "two", true],
      null: null,
    });
  });
});

describe("buildJsonToolSystemPrompt", () => {
  it("returns empty string with no tools or prompt", () => {
    expect(buildJsonToolSystemPrompt(undefined, [])).toBe("");
  });

  it("returns original prompt when no tools", () => {
    const prompt = "You are helpful.";
    expect(buildJsonToolSystemPrompt(prompt, [])).toBe(prompt);
  });

  it("generates tool instructions with single tool", () => {
    const tools: ToolDefinition[] = [
      {
        name: "get_weather",
        description: "Get weather info",
        parameters: {
          type: "object",
          properties: { city: { type: "string" } },
        },
      },
    ];

    const result = buildJsonToolSystemPrompt(undefined, tools);
    expect(result).toContain("get_weather");
    expect(result).toContain("Get weather info");
    expect(result).toContain("Available Tools");
    expect(result).toContain("```tool_call");
    expect(result).toContain("Only request one tool call at a time");
  });

  it("includes all tools in prompt", () => {
    const tools: ToolDefinition[] = [
      {
        name: "search",
        description: "Search",
        parameters: { type: "object", properties: {} },
      },
      {
        name: "calculate",
        description: "Calculate",
        parameters: { type: "object", properties: {} },
      },
    ];

    const result = buildJsonToolSystemPrompt(undefined, tools);
    expect(result).toContain("search");
    expect(result).toContain("calculate");
  });

  it("formats tools as valid JSON", () => {
    const tools: ToolDefinition[] = [
      {
        name: "tool1",
        description: "First",
        parameters: { type: "object", properties: {} },
      },
    ];

    const result = buildJsonToolSystemPrompt(undefined, tools);
    const jsonMatch = result.match(/\[\s*{[\s\S]*}\s*\]/);
    expect(jsonMatch).toBeTruthy();
    const parsed = JSON.parse(jsonMatch![0]);
    expect(parsed[0].name).toBe("tool1");
  });

  it("uses default description when missing", () => {
    const tools: ToolDefinition[] = [
      {
        name: "test",
        description: undefined as any,
        parameters: { type: "object", properties: {} },
      },
    ];

    const result = buildJsonToolSystemPrompt(undefined, tools);
    expect(result).toContain("No description provided.");
  });

  it("appends instructions to existing system prompt", () => {
    const prompt = "You are helpful.";
    const tools: ToolDefinition[] = [
      {
        name: "test",
        description: "Test",
        parameters: { type: "object", properties: {} },
      },
    ];

    const result = buildJsonToolSystemPrompt(prompt, tools);
    expect(result).toContain("You are helpful.");
    expect(result).toContain("Available Tools");
    expect(result.indexOf("You are helpful.")).toBeLessThan(
      result.indexOf("Available Tools"),
    );
  });

  it("includes tool calling format and instructions", () => {
    const tools: ToolDefinition[] = [
      {
        name: "test",
        description: "Test",
        parameters: { type: "object", properties: {} },
      },
    ];

    const result = buildJsonToolSystemPrompt(undefined, tools);
    expect(result).toContain('"name": "tool_name"');
    expect(result).toContain('"arguments"');
    expect(result).toContain("```tool_result");
    expect(result).toContain("Use exact tool and parameter names");
  });
});

describe("extractSystemPrompt", () => {
  it("extracts system prompt and removes it from messages", () => {
    const messages = [
      { role: "system", content: "You are helpful." },
      { role: "user", content: "Hello" },
    ] as any;

    const result = extractSystemPrompt(messages);
    expect(result.systemPrompt).toBe("You are helpful.");
    expect(result.messages).toHaveLength(1);
    expect(result.messages[0].role).toBe("user");
  });

  it("returns undefined when no system message exists", () => {
    const messages = [
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi!" },
    ] as any;

    const result = extractSystemPrompt(messages);
    expect(result.systemPrompt).toBeUndefined();
    expect(result.messages).toEqual(messages);
  });

  it("handles empty messages array", () => {
    const messages: any = [];

    const result = extractSystemPrompt(messages);
    expect(result.systemPrompt).toBeUndefined();
    expect(result.messages).toEqual([]);
  });

  it("extracts system message from any position", () => {
    const messages = [
      { role: "user", content: "Hello" },
      { role: "system", content: "Be concise." },
      { role: "assistant", content: "Hi!" },
    ] as any;

    const result = extractSystemPrompt(messages);
    expect(result.systemPrompt).toBe("Be concise.");
    expect(result.messages).toHaveLength(2);
    expect(result.messages[0].role).toBe("user");
    expect(result.messages[1].role).toBe("assistant");
  });

  it("combines multiple system messages", () => {
    const messages = [
      { role: "system", content: "Be helpful." },
      { role: "user", content: "Hello" },
      { role: "system", content: "Be concise." },
    ] as any;

    const result = extractSystemPrompt(messages);
    expect(result.systemPrompt).toBe("Be helpful.\n\nBe concise.");
    expect(result.messages).toHaveLength(1);
    expect(result.messages[0].role).toBe("user");
  });
});
