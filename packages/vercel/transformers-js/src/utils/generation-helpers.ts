import type { ToolDefinition } from "@browser-ai/shared";
import { convertToolsToHuggingFaceFormat } from "./convert-tools";

export function buildApplyChatTemplateOptions({
  tools,
  enableThinking = false,
}: {
  tools?: ToolDefinition[];
  enableThinking?: boolean;
}): Record<string, unknown> {
  const hfTools = tools?.length
    ? convertToolsToHuggingFaceFormat(tools)
    : undefined;

  return {
    add_generation_prompt: true,
    ...(hfTools ? { tools: hfTools } : {}),
    ...(enableThinking ? { enable_thinking: true } : {}),
  };
}

export function normalizeStreamedTextChunk(output: string): string | null {
  const trimmed = output.trim();
  const isSpecialToken =
    /^<\|[^>]+>$/.test(trimmed) || /^<[^|>]+\|>$/.test(trimmed);
  const isToolCallToken =
    trimmed === "<|tool_call|>" ||
    trimmed === "<|tool_call>" ||
    trimmed === "<tool_call|>" ||
    /^<\/?tool_call>$/.test(trimmed);

  if (isSpecialToken && !isToolCallToken && !trimmed.includes("channel")) {
    return null;
  }

  if (trimmed === "<|channel>") return "<think>";
  if (trimmed === "<channel|>") return "</think>";

  return output;
}
