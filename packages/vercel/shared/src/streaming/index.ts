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
} from "./tool-call-detector";
export {
  createArgumentsStreamState,
  extractArgumentsDelta,
  extractToolName,
  type ArgumentsStreamState,
} from "./tool-call-stream-utils";
