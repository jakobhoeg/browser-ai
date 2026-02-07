import { describe, it, expect } from "vitest";
import { parseJsonFunctionCalls } from "@browser-ai/shared";

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
