import type { ToolResult } from "../types";

/**
 * Builds a JSON-serializable payload for a single tool result.
 * Includes tool name, result data, error flag, and optional call ID.
 *
 * @param result - The tool execution result to format
 * @returns Object containing formatted result data ready for JSON serialization
 */
function buildResultPayload(result: ToolResult): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    name: result.toolName,
    result: result.result ?? null,
    error: Boolean(result.isError),
  };

  if (result.toolCallId) {
    payload.id = result.toolCallId;
  }

  return payload;
}

/**
 * Formats tool execution results as JSON for continuation in the conversation.
 *
 * Each result is serialized as a single JSON object. Multiple results (for parallel
 * execution scenarios) are emitted on separate lines within a ```tool_result code fence.
 *
 * @param results - Array of tool execution results to format
 * @returns Formatted string with results in tool_result code fence, or empty string if no results
 * @example
 * ```typescript
 * formatToolResults([
 *   { toolCallId: "call_123", toolName: "search", result: { data: "..." } }
 * ])
 * // Returns: ```tool_result\n{"id":"call_123","name":"search","result":{...},"error":false}\n```
 * ```
 */
export function formatToolResults(results: ToolResult[]): string {
  if (!results || results.length === 0) {
    return "";
  }

  const payloads = results.map((result) =>
    JSON.stringify(buildResultPayload(result)),
  );

  return `\`\`\`tool_result
${payloads.join("\n")}
\`\`\``;
}

/**
 * Formats a single tool result.
 * Convenience wrapper around formatToolResults for single result scenarios.
 *
 * @param result - The tool execution result to format
 * @returns Formatted string with result in tool_result code fence
 */
export function formatSingleToolResult(result: ToolResult): string {
  return formatToolResults([result]);
}
