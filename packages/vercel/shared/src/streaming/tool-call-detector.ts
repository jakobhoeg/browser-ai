/**
 * ToolCallFenceDetector - Detects and extracts tool call fences from streaming text
 *
 * This module handles the complex task of detecting tool call fences in a stream
 * where fences might be split across multiple chunks. It uses overlap detection
 * to avoid emitting text that might be the beginning of a fence.
 */

/**
 * Result of fence detection operation
 */
export interface FenceDetectionResult {
  fence: string | null;
  prefixText: string;
  remainingText: string;
  /** Length of potential partial fence at buffer end */
  overlapLength: number;
}

/**
 * Result of streaming fence content detection
 */
export interface StreamingFenceResult {
  inFence: boolean;
  /** Content that can be safely emitted (either as text or tool-input-delta) */
  safeContent: string;
  completeFence: string | null;
  textAfterFence: string;
}

/**
 * Fence pattern configuration
 */
export interface FencePattern {
  start: string;
  end: string;
  reconstructStart: string;
  isRegex?: boolean;
  matchedStart?: string;
}

/**
 * Options for configuring the ToolCallFenceDetector
 */
export interface ToolCallFenceDetectorOptions {
  /** Custom fence patterns to use instead of defaults */
  patterns?: FencePattern[];
  /** Enable Python-style function call detection: [functionName(args)] */
  enablePythonStyle?: boolean;
}

/**
 * Default fence patterns for tool call detection
 */
export const DEFAULT_FENCE_PATTERNS: FencePattern[] = [
  { start: "```tool_call", end: "```", reconstructStart: "```tool_call\n" },
  { start: "```tool-call", end: "```", reconstructStart: "```tool-call\n" },
];

/**
 * Extended fence patterns including XML-style tags
 */
export const EXTENDED_FENCE_PATTERNS: FencePattern[] = [
  ...DEFAULT_FENCE_PATTERNS,
  {
    start: "<tool_call>",
    end: "</tool_call>",
    reconstructStart: "<tool_call>",
  },
];

/**
 * Detects tool call fences in streaming text with support for partial matches
 */
export class ToolCallFenceDetector {
  private readonly fencePatterns: FencePattern[];
  private readonly enablePythonStyle: boolean;
  private readonly pythonStyleRegex = /\[(\w+)\(/g;
  private readonly fenceStarts: string[];
  private buffer = "";

  private inFence = false;
  private fenceStartBuffer = ""; // Accumulated fence content
  private currentFencePattern: FencePattern | null = null;

  constructor(options: ToolCallFenceDetectorOptions = {}) {
    this.fencePatterns = options.patterns ?? EXTENDED_FENCE_PATTERNS;
    this.enablePythonStyle = options.enablePythonStyle ?? true;
    this.fenceStarts = this.fencePatterns.map((p) => p.start);
  }

  addChunk(chunk: string): void {
    this.buffer += chunk;
  }

  getBuffer(): string {
    return this.buffer;
  }

  clearBuffer(): void {
    this.buffer = "";
  }

  /**
   * Detects if there's a complete fence in the buffer
   * @returns Detection result with fence info and safe text
   */
  detectFence(): FenceDetectionResult {
    const {
      index: startIdx,
      prefix: matchedPrefix,
      pattern,
    } = this.findFenceStart(this.buffer);

    // No fence start found
    if (startIdx === -1) {
      // Compute how much of the buffer end might be a partial fence start
      const overlap = this.computeOverlapLength(this.buffer, this.fenceStarts);
      const safeTextLength = this.buffer.length - overlap;

      const prefixText =
        safeTextLength > 0 ? this.buffer.slice(0, safeTextLength) : "";
      const remaining = overlap > 0 ? this.buffer.slice(-overlap) : "";

      // Update buffer to keep only the overlap
      this.buffer = remaining;

      return {
        fence: null,
        prefixText,
        remainingText: "",
        overlapLength: overlap,
      };
    }

    const prefixText = this.buffer.slice(0, startIdx);
    this.buffer = this.buffer.slice(startIdx);

    // Look for closing fence using the matched pattern's end marker
    const prefixLength = matchedPrefix?.length ?? 0;
    const fenceEnd = pattern?.end ?? "```";
    const closingIdx = this.buffer.indexOf(fenceEnd, prefixLength);

    // Fence not complete yet
    if (closingIdx === -1) {
      // Keep the buffer as-is, waiting for more data
      return {
        fence: null,
        prefixText,
        remainingText: "",
        overlapLength: 0,
      };
    }

    // Complete fence found!
    const endPos = closingIdx + fenceEnd.length;
    const fence = this.buffer.slice(0, endPos);
    const remainingText = this.buffer.slice(endPos);

    // Clear the buffer since we extracted everything
    this.buffer = "";

    return {
      fence,
      prefixText,
      remainingText,
      overlapLength: 0,
    };
  }

  /**
   * Finds the first occurrence of any fence start marker
   *
   * @param text - Text to search in
   * @returns Index of first fence start and which pattern matched
   * @private
   */
  private findFenceStart(text: string): {
    index: number;
    prefix: string | null;
    pattern: FencePattern | null;
  } {
    let bestIndex = -1;
    let matchedPrefix: string | null = null;
    let matchedPattern: FencePattern | null = null;

    for (const pattern of this.fencePatterns) {
      const idx = text.indexOf(pattern.start);
      if (idx !== -1 && (bestIndex === -1 || idx < bestIndex)) {
        bestIndex = idx;
        matchedPrefix = pattern.start;
        matchedPattern = pattern;
      }
    }

    if (this.enablePythonStyle) {
      this.pythonStyleRegex.lastIndex = 0;
      const pythonMatch = this.pythonStyleRegex.exec(text);
      if (pythonMatch && (bestIndex === -1 || pythonMatch.index < bestIndex)) {
        bestIndex = pythonMatch.index;
        matchedPrefix = pythonMatch[0];
        matchedPattern = {
          start: pythonMatch[0],
          end: ")]",
          reconstructStart: pythonMatch[0],
          isRegex: true,
        };
      }
    }

    return { index: bestIndex, prefix: matchedPrefix, pattern: matchedPattern };
  }

  /**
   * Computes the maximum overlap between the end of text and the start of any prefix
   * @param text - Text to check for overlap
   * @param prefixes - List of prefixes to check against
   * @returns Length of the maximum overlap found
   */
  private computeOverlapLength(text: string, prefixes: string[]): number {
    let overlap = 0;

    for (const prefix of prefixes) {
      const maxLength = Math.min(text.length, prefix.length - 1);

      for (let size = maxLength; size > 0; size -= 1) {
        // Check if the last 'size' characters of text match the first 'size' characters of prefix
        if (prefix.startsWith(text.slice(-size))) {
          overlap = Math.max(overlap, size);
          break;
        }
      }
    }

    return overlap;
  }

  /**
   * Checks if the buffer currently contains any text
   */
  hasContent(): boolean {
    return this.buffer.length > 0;
  }

  /**
   * Gets the buffer size
   */
  getBufferSize(): number {
    return this.buffer.length;
  }

  /**
   * Detect and stream fence content in real-time for true incremental streaming
   * @returns Streaming result with current state and safe content to emit
   */
  detectStreamingFence(): StreamingFenceResult {
    if (!this.inFence) {
      // Look for fence start
      const {
        index: startIdx,
        prefix: matchedPrefix,
        pattern,
      } = this.findFenceStart(this.buffer);

      if (startIdx === -1) {
        // No fence start found - emit safe text
        const overlap = this.computeOverlapLength(
          this.buffer,
          this.fenceStarts,
        );
        const safeTextLength = this.buffer.length - overlap;
        const safeContent =
          safeTextLength > 0 ? this.buffer.slice(0, safeTextLength) : "";
        this.buffer = this.buffer.slice(safeTextLength);

        return {
          inFence: false,
          safeContent,
          completeFence: null,
          textAfterFence: "",
        };
      }

      // Found fence start!
      const prefixText = this.buffer.slice(0, startIdx);
      const fenceStartLength = matchedPrefix?.length ?? 0;

      // Move buffer past the fence start marker
      this.buffer = this.buffer.slice(startIdx + fenceStartLength);

      if (
        pattern &&
        pattern.start.startsWith("```") &&
        this.buffer.startsWith("\n")
      ) {
        this.buffer = this.buffer.slice(1);
      }

      this.inFence = true;
      this.fenceStartBuffer = "";
      this.currentFencePattern = pattern;

      return {
        inFence: true,
        safeContent: prefixText, // Emit any text before the fence
        completeFence: null,
        textAfterFence: "",
      };
    }

    // We're inside a fence - look for fence end using the current pattern
    const fenceEnd = this.currentFencePattern?.end ?? "```";
    const closingIdx = this.buffer.indexOf(fenceEnd);

    if (closingIdx === -1) {
      // No fence end yet - emit safe content (leaving potential fence end marker)
      const overlap = this.computeOverlapLength(this.buffer, [fenceEnd]);
      const safeContentLength = this.buffer.length - overlap;

      if (safeContentLength > 0) {
        const safeContent = this.buffer.slice(0, safeContentLength);
        this.fenceStartBuffer += safeContent;
        this.buffer = this.buffer.slice(safeContentLength);

        return {
          inFence: true,
          safeContent,
          completeFence: null,
          textAfterFence: "",
        };
      }

      // Nothing safe to emit yet
      return {
        inFence: true,
        safeContent: "",
        completeFence: null,
        textAfterFence: "",
      };
    }

    // Found fence end!
    const fenceContent = this.buffer.slice(0, closingIdx);
    this.fenceStartBuffer += fenceContent;

    // Reconstruct complete fence using the current pattern
    const reconstructStart =
      this.currentFencePattern?.reconstructStart ?? "```tool_call\n";
    const completeFence = `${reconstructStart}${this.fenceStartBuffer}${fenceEnd}`;

    // Get text after fence
    const textAfterFence = this.buffer.slice(closingIdx + fenceEnd.length);

    // Reset state
    this.inFence = false;
    this.fenceStartBuffer = "";
    this.currentFencePattern = null;
    this.buffer = textAfterFence;

    return {
      inFence: false,
      safeContent: fenceContent, // Emit the last bit of fence content
      completeFence,
      textAfterFence,
    };
  }

  isInFence(): boolean {
    return this.inFence;
  }

  resetStreamingState(): void {
    this.inFence = false;
    this.fenceStartBuffer = "";
    this.currentFencePattern = null;
  }
}

/**
 * Creates a basic ToolCallFenceDetector with default markdown fence patterns
 */
export function createBasicDetector(): ToolCallFenceDetector {
  return new ToolCallFenceDetector({
    patterns: DEFAULT_FENCE_PATTERNS,
    enablePythonStyle: false,
  });
}

/**
 * Creates an extended ToolCallFenceDetector with all fence patterns
 */
export function createExtendedDetector(): ToolCallFenceDetector {
  return new ToolCallFenceDetector({
    patterns: EXTENDED_FENCE_PATTERNS,
    enablePythonStyle: true,
  });
}
