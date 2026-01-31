import { describe, it, expect } from "vitest";
import {
  isFunctionTool,
  createUnsupportedSettingWarning,
  createUnsupportedToolWarning,
  buildJsonToolSystemPrompt,
  parseJsonFunctionCalls,
  hasJsonFunctionCalls,
  extractJsonFunctionCallsBlock,
  formatToolResults,
  formatSingleToolResult,
  ToolCallFenceDetector,
  createBasicDetector,
  createExtendedDetector,
  DEFAULT_FENCE_PATTERNS,
  EXTENDED_FENCE_PATTERNS,
} from "../src";

describe("@browser-ai/shared exports", () => {
  describe("isFunctionTool", () => {
    it("should return true for function tools", () => {
      const functionTool = {
        type: "function" as const,
        name: "test",
        inputSchema: { type: "object" as const },
      };
      expect(isFunctionTool(functionTool)).toBe(true);
    });

    it("should return false for provider tools", () => {
      const providerTool = {
        type: "provider",
        name: "test",
        id: "test.provider",
        args: {},
      } as const;
      expect(isFunctionTool(providerTool)).toBe(false);
    });
  });

  describe("createUnsupportedSettingWarning", () => {
    it("should create a warning object", () => {
      const warning = createUnsupportedSettingWarning(
        "maxTokens",
        "Not supported",
      );
      expect(warning).toEqual({
        type: "unsupported",
        feature: "maxTokens",
        details: "Not supported",
      });
    });
  });

  describe("createUnsupportedToolWarning", () => {
    it("should create a warning object with tool name", () => {
      const tool = {
        type: "provider",
        name: "customTool",
        id: "custom.tool",
        args: {},
      } as const;
      const warning = createUnsupportedToolWarning(tool, "Not supported");
      expect(warning).toEqual({
        type: "unsupported",
        feature: "tool:customTool",
        details: "Not supported",
      });
    });
  });

  describe("buildJsonToolSystemPrompt", () => {
    it("should return empty string for no tools", () => {
      const result = buildJsonToolSystemPrompt(undefined, []);
      expect(result).toBe("");
    });

    it("should return original prompt for no tools", () => {
      const result = buildJsonToolSystemPrompt("Hello", []);
      expect(result).toBe("Hello");
    });

    it("should build prompt with tools", () => {
      const result = buildJsonToolSystemPrompt(undefined, [
        {
          name: "test",
          description: "Test tool",
          parameters: { type: "object" },
        },
      ]);
      expect(result).toContain("Available Tools");
      expect(result).toContain("test");
    });
  });

  describe("parseJsonFunctionCalls", () => {
    it("should return empty for no tool calls", () => {
      const result = parseJsonFunctionCalls("Hello world");
      expect(result.toolCalls).toHaveLength(0);
      expect(result.textContent).toBe("Hello world");
    });

    it("should parse tool call fences", () => {
      const input = '```tool_call\n{"name": "test", "arguments": {}}\n```';
      const result = parseJsonFunctionCalls(input);
      expect(result.toolCalls).toHaveLength(1);
      expect(result.toolCalls[0].toolName).toBe("test");
    });

    it("should parse XML-style tool calls", () => {
      const input =
        '<tool_call>{"name": "search", "arguments": {"q": "test"}}</tool_call>';
      const result = parseJsonFunctionCalls(input);
      expect(result.toolCalls[0].toolName).toBe("search");
      expect(result.toolCalls[0].args).toEqual({ q: "test" });
    });

    it("should parse Python-style tool calls", () => {
      const result = parseJsonFunctionCalls('[search(query="hello")]');
      expect(result.toolCalls[0].toolName).toBe("search");
      expect(result.toolCalls[0].args).toEqual({ query: "hello" });
    });

    it("should parse array of tool calls", () => {
      const input = '```tool_call\n[{"name": "a"}, {"name": "b"}]\n```';
      const result = parseJsonFunctionCalls(input);
      expect(result.toolCalls).toHaveLength(2);
    });

    it("should support parameters field", () => {
      const input =
        '```tool_call\n{"name": "test", "parameters": {"x": 1}}\n```';
      const result = parseJsonFunctionCalls(input);
      expect(result.toolCalls[0].args).toEqual({ x: 1 });
    });

    it("should preserve text content around tool calls", () => {
      const input = 'Before ```tool_call\n{"name": "t"}\n``` After';
      const result = parseJsonFunctionCalls(input);
      expect(result.textContent).toContain("Before");
      expect(result.textContent).toContain("After");
    });
  });

  describe("hasJsonFunctionCalls", () => {
    it("should return true when tool calls present", () => {
      expect(hasJsonFunctionCalls("```tool_call\n{}\n```")).toBe(true);
      expect(hasJsonFunctionCalls("<tool_call>{}</tool_call>")).toBe(true);
    });

    it("should return false when no tool calls", () => {
      expect(hasJsonFunctionCalls("plain text")).toBe(false);
    });
  });

  describe("extractJsonFunctionCallsBlock", () => {
    it("should extract first fence block", () => {
      const input = 'text ```tool_call\n{"name": "test"}\n``` more';
      const block = extractJsonFunctionCallsBlock(input);
      expect(block).toBe('```tool_call\n{"name": "test"}\n```');
    });

    it("should return null when no fence", () => {
      expect(extractJsonFunctionCallsBlock("no fence here")).toBeNull();
    });
  });

  describe("formatToolResults", () => {
    it("should return empty string for no results", () => {
      expect(formatToolResults([])).toBe("");
    });

    it("should format tool results", () => {
      const result = formatToolResults([
        { toolCallId: "1", toolName: "test", result: { data: "ok" } },
      ]);
      expect(result).toContain("tool_result");
      expect(result).toContain("test");
    });

    it("should format multiple results", () => {
      const result = formatToolResults([
        { toolCallId: "1", toolName: "a", result: "x" },
        { toolCallId: "2", toolName: "b", result: "y" },
      ]);
      expect(result).toContain('"name":"a"');
      expect(result).toContain('"name":"b"');
    });

    it("should include error flag", () => {
      const result = formatToolResults([
        { toolCallId: "1", toolName: "t", result: "err", isError: true },
      ]);
      expect(result).toContain('"error":true');
    });
  });

  describe("formatSingleToolResult", () => {
    it("should format a single result", () => {
      const result = formatSingleToolResult({
        toolCallId: "123",
        toolName: "search",
        result: { items: [] },
      });
      expect(result).toContain("tool_result");
      expect(result).toContain("search");
    });
  });

  describe("ToolCallFenceDetector", () => {
    it("should create instance", () => {
      const detector = new ToolCallFenceDetector();
      expect(detector).toBeInstanceOf(ToolCallFenceDetector);
    });

    it("should detect complete fence", () => {
      const detector = new ToolCallFenceDetector();
      detector.addChunk('```tool_call\n{"name": "test"}\n```');
      const result = detector.detectFence();
      expect(result.fence).not.toBeNull();
    });

    it("should handle chunked input", () => {
      const detector = new ToolCallFenceDetector();
      detector.addChunk("```tool_call\n");
      detector.addChunk('{"name": "test"}');
      detector.addChunk("\n```");
      expect(detector.detectFence().fence).not.toBeNull();
    });

    it("should return prefix text before fence", () => {
      const detector = new ToolCallFenceDetector();
      detector.addChunk('Hello ```tool_call\n{"name": "t"}\n```');
      const result = detector.detectFence();
      expect(result.prefixText).toBe("Hello ");
    });

    it("should track buffer state", () => {
      const detector = new ToolCallFenceDetector();
      expect(detector.hasContent()).toBe(false);
      detector.addChunk("test");
      expect(detector.hasContent()).toBe(true);
      expect(detector.getBufferSize()).toBe(4);
    });

    it("should clear buffer", () => {
      const detector = new ToolCallFenceDetector();
      detector.addChunk("test");
      detector.clearBuffer();
      expect(detector.getBuffer()).toBe("");
    });

    it("should detect streaming fence", () => {
      const detector = new ToolCallFenceDetector();
      detector.addChunk('```tool_call\n{"name": "t"}\n```');
      const result = detector.detectStreamingFence();
      expect(result.inFence).toBe(true);
    });
  });

  describe("createBasicDetector", () => {
    it("should create detector with default patterns", () => {
      const detector = createBasicDetector();
      expect(detector).toBeInstanceOf(ToolCallFenceDetector);
      detector.addChunk('```tool_call\n{"name": "t"}\n```');
      expect(detector.detectFence().fence).not.toBeNull();
    });
  });

  describe("createExtendedDetector", () => {
    it("should create detector with extended patterns", () => {
      const detector = createExtendedDetector();
      detector.addChunk('<tool_call>{"name": "t"}</tool_call>');
      expect(detector.detectFence().fence).not.toBeNull();
    });
  });

  describe("fence patterns", () => {
    it("DEFAULT_FENCE_PATTERNS should have markdown patterns", () => {
      expect(DEFAULT_FENCE_PATTERNS.length).toBeGreaterThan(0);
      expect(DEFAULT_FENCE_PATTERNS.some((p) => p.start.includes("```"))).toBe(
        true,
      );
    });

    it("EXTENDED_FENCE_PATTERNS should include XML tags", () => {
      expect(EXTENDED_FENCE_PATTERNS.length).toBeGreaterThan(
        DEFAULT_FENCE_PATTERNS.length,
      );
      expect(
        EXTENDED_FENCE_PATTERNS.some((p) => p.start.includes("<tool_call>")),
      ).toBe(true);
    });
  });
});
