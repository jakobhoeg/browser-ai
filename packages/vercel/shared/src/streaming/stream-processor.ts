import type { LanguageModelV3StreamPart } from "@ai-sdk/provider";
import { ToolCallFenceDetector } from "./tool-call-detector";
import {
  createArgumentsStreamState,
  extractArgumentsDelta,
  extractToolName,
} from "./tool-call-stream-utils";
import { parseJsonFunctionCalls } from "../tool-calling/parse-json-function-calls";
import type { ParsedToolCall } from "../types";

export interface ToolCallStreamResult {
  toolCallDetected: boolean;
  toolCalls: ParsedToolCall[];
  /** Text appearing after the tool call fence — caller decides when to emit it */
  trailingText: string;
}

export function generateToolCallId(): string {
  return `call_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

/**
 * Processes an async iterable of string chunks, detecting tool call fences and
 * emitting the appropriate stream events via `controller` and `emitTextDelta`.
 *
 * When `stopEarlyOnToolCall` is true the function stops consuming chunks as soon as
 * a tool call is detected (useful when the caller needs to cancel the source).
 * When false (the default) the function continues draining remaining chunks without
 * processing them, allowing the underlying stream/engine to conclude normally.
 */
export async function processToolCallStream(
  chunks: AsyncIterable<string>,
  emitTextDelta: (delta: string) => void,
  controller: ReadableStreamDefaultController<LanguageModelV3StreamPart>,
  options?: { stopEarlyOnToolCall?: boolean },
): Promise<ToolCallStreamResult> {
  const fenceDetector = new ToolCallFenceDetector();

  let currentToolCallId: string | null = null;
  let toolInputStartEmitted = false;
  let accumulatedFenceContent = "";
  let argumentsStreamState = createArgumentsStreamState();
  let insideFence = false;

  let toolCallDetected = false;
  let toolCalls: ParsedToolCall[] = [];
  let trailingText = "";

  const resetFenceState = () => {
    currentToolCallId = null;
    toolInputStartEmitted = false;
    accumulatedFenceContent = "";
    argumentsStreamState = createArgumentsStreamState();
    insideFence = false;
  };

  for await (const chunk of chunks) {
    if (toolCallDetected) {
      // Drain without processing so the underlying stream/engine can conclude.
      continue;
    }

    fenceDetector.addChunk(chunk);

    while (fenceDetector.hasContent()) {
      const wasInsideFence = insideFence;
      const result = fenceDetector.detectStreamingFence();
      insideFence = result.inFence;

      let madeProgress = false;

      if (!wasInsideFence && result.inFence) {
        if (result.safeContent) {
          emitTextDelta(result.safeContent);
          madeProgress = true;
        }

        currentToolCallId = generateToolCallId();
        toolInputStartEmitted = false;
        accumulatedFenceContent = "";
        argumentsStreamState = createArgumentsStreamState();
        insideFence = true;

        continue;
      }

      if (result.completeFence) {
        madeProgress = true;
        if (result.safeContent) {
          accumulatedFenceContent += result.safeContent;
        }

        if (toolInputStartEmitted && currentToolCallId) {
          const delta = extractArgumentsDelta(
            accumulatedFenceContent,
            argumentsStreamState,
          );
          if (delta.length > 0) {
            controller.enqueue({
              type: "tool-input-delta",
              id: currentToolCallId,
              delta,
            });
          }
        }

        const parsed = parseJsonFunctionCalls(result.completeFence);
        const selectedToolCalls = parsed.toolCalls.slice(0, 1);

        if (selectedToolCalls.length === 0) {
          emitTextDelta(result.completeFence);
          if (result.textAfterFence) {
            emitTextDelta(result.textAfterFence);
          }
          resetFenceState();
          continue;
        }

        if (currentToolCallId) {
          selectedToolCalls[0].toolCallId = currentToolCallId;
        }

        for (const [index, call] of selectedToolCalls.entries()) {
          const toolCallId =
            index === 0 && currentToolCallId
              ? currentToolCallId
              : call.toolCallId;
          const toolName = call.toolName;
          const argsJson = JSON.stringify(call.args ?? {});

          if (toolCallId === currentToolCallId) {
            if (!toolInputStartEmitted) {
              controller.enqueue({
                type: "tool-input-start",
                id: toolCallId,
                toolName,
              });
              toolInputStartEmitted = true;
            }

            const delta = extractArgumentsDelta(
              accumulatedFenceContent,
              argumentsStreamState,
            );
            if (delta.length > 0) {
              controller.enqueue({
                type: "tool-input-delta",
                id: toolCallId,
                delta,
              });
            }
          } else {
            controller.enqueue({
              type: "tool-input-start",
              id: toolCallId,
              toolName,
            });
            if (argsJson.length > 0) {
              controller.enqueue({
                type: "tool-input-delta",
                id: toolCallId,
                delta: argsJson,
              });
            }
          }

          controller.enqueue({ type: "tool-input-end", id: toolCallId });
          controller.enqueue({
            type: "tool-call",
            toolCallId,
            toolName,
            input: argsJson,
            providerExecuted: false,
          });
        }

        trailingText = result.textAfterFence ?? "";
        toolCalls = selectedToolCalls;
        toolCallDetected = true;
        resetFenceState();
        break; // stop processing inner buffer
      }

      if (insideFence) {
        if (result.safeContent) {
          accumulatedFenceContent += result.safeContent;
          madeProgress = true;

          const toolName = extractToolName(accumulatedFenceContent);
          if (toolName && !toolInputStartEmitted && currentToolCallId) {
            controller.enqueue({
              type: "tool-input-start",
              id: currentToolCallId,
              toolName,
            });
            toolInputStartEmitted = true;
          }

          if (toolInputStartEmitted && currentToolCallId) {
            const delta = extractArgumentsDelta(
              accumulatedFenceContent,
              argumentsStreamState,
            );
            if (delta.length > 0) {
              controller.enqueue({
                type: "tool-input-delta",
                id: currentToolCallId,
                delta,
              });
            }
          }
        }

        continue;
      }

      if (!insideFence && result.safeContent) {
        emitTextDelta(result.safeContent);
        madeProgress = true;
      }

      if (!madeProgress) {
        break;
      }
    }

    if (toolCallDetected && options?.stopEarlyOnToolCall) {
      break; // caller will cancel/drain the underlying source
    }
  }

  // Flush any remaining buffer when no tool call was detected
  if (!toolCallDetected && fenceDetector.hasContent()) {
    emitTextDelta(fenceDetector.getBuffer());
    fenceDetector.clearBuffer();
  }

  return { toolCallDetected, toolCalls, trailingText };
}
