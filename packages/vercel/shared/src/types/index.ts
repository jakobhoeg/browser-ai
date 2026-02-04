export type {
  JSONSchema,
  ToolDefinition,
  ParsedToolCall,
  ToolResult,
  ParsedResponse,
} from "./tool-calling";

/**
 * Callback type for receiving model download/initialization progress updates.
 * @param progress - A value between 0 and 1 indicating completion.
 */
export type DownloadProgressCallback = (progress: number) => void;
