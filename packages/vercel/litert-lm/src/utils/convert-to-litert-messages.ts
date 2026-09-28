import {
  LanguageModelV4Prompt,
  LanguageModelV4ToolResultPart,
  LanguageModelV4ToolResultOutput,
  UnsupportedFunctionalityError,
} from "@ai-sdk/provider";
import { formatToolResults, type ToolResult } from "@browser-ai/shared";

/**
 * The simplified message shape accepted by the LiteRT-LM Web `Conversation`
 * API. The web SDK is currently text-in / text-out, so every part collapses
 * to a plain `{ role, content }` pair.
 */
export type LiteRTMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

function convertToolResultOutput(output: LanguageModelV4ToolResultOutput): {
  value: unknown;
  isError: boolean;
} {
  switch (output.type) {
    case "text":
      return { value: output.value, isError: false };
    case "json":
      return { value: output.value, isError: false };
    case "error-text":
      return { value: output.value, isError: true };
    case "error-json":
      return { value: output.value, isError: true };
    case "content":
      return { value: output.value, isError: false };
    case "execution-denied":
      return { value: output.reason, isError: true };
    default: {
      const exhaustiveCheck: never = output;
      return { value: exhaustiveCheck, isError: false };
    }
  }
}

function toToolResult(part: LanguageModelV4ToolResultPart): ToolResult {
  const { value, isError } = convertToolResultOutput(part.output);
  return {
    toolCallId: part.toolCallId,
    toolName: part.toolName,
    result: value,
    isError,
  };
}

/**
 * Converts an AI SDK `LanguageModelV4Prompt` into the message list expected by
 * the LiteRT-LM Web SDK.
 *
 * Notes:
 * - System prompts are passed through verbatim.
 * - Multi-turn history is preserved so context survives the (stateless) AI SDK
 *   call. The calling code puts everything except the final message into the
 *   conversation `preface`, then sends the final message.
 * - Tool results are serialized to text and injected as a user turn (the web
 *   SDK has no native tool-calling yet). Tool-call parts on assistant messages
 *   are dropped (a warning is emitted upstream).
 * - Files/images are not supported by the text-only web runtime and throw.
 */
export function convertToLiteRTMessages(
  prompt: LanguageModelV4Prompt,
): LiteRTMessage[] {
  const messages: LiteRTMessage[] = [];

  for (const message of prompt) {
    switch (message.role) {
      case "system":
        messages.push({ role: "system", content: message.content });
        break;

      case "user": {
        const filePart = message.content.find((part) => part.type === "file");
        if (filePart && filePart.type === "file") {
          throw new UnsupportedFunctionalityError({
            functionality: `file input with media type '${filePart.mediaType}' (LiteRT-LM Web only supports text)`,
          });
        }

        const text = message.content
          .filter((part) => part.type === "text")
          .map((part) => (part as { text: string }).text)
          .join("\n");

        messages.push({ role: "user", content: text });
        break;
      }

      case "assistant": {
        const text = message.content
          .filter((part) => part.type === "text")
          .map((part) => (part as { text: string }).text)
          .join("\n");

        // Tool-call parts are not natively supported by the web SDK yet.
        if (text) messages.push({ role: "assistant", content: text });
        break;
      }

      case "tool": {
        const toolResults: ToolResult[] = message.content
          .filter((part) => part.type === "tool-result")
          .map(toToolResult);

        messages.push({
          role: "user",
          content: formatToolResults(toolResults),
        });
        break;
      }
    }
  }

  return messages;
}
