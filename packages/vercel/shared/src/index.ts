// Types
export type {
  JSONSchema,
  ToolDefinition,
  ParsedToolCall,
  ToolResult,
  ParsedResponse,
} from "./types";

// Utils
export { isFunctionTool } from "./utils/tool-utils";
export {
  createUnsupportedSettingWarning,
  createUnsupportedToolWarning,
} from "./utils/warnings";

// Tool Calling
export { buildJsonToolSystemPrompt } from "./tool-calling/build-json-system-prompt";
export {
  formatToolResults,
  formatSingleToolResult,
} from "./tool-calling/format-tool-results";
export {
  parseJsonFunctionCalls,
  hasJsonFunctionCalls,
  extractJsonFunctionCallsBlock,
  type ParseJsonFunctionCallsOptions,
} from "./tool-calling/parse-json-function-calls";

// Streaming
export {
  ToolCallFenceDetector,
  createBasicDetector,
  createExtendedDetector,
  DEFAULT_FENCE_PATTERNS,
  EXTENDED_FENCE_PATTERNS,
  type FenceDetectionResult,
  type StreamingFenceResult,
  type FencePattern,
  type ToolCallFenceDetectorOptions,
} from "./streaming";
