import type { ParsedResponse, ParsedToolCall } from "../types";

/**
 * Options for configuring the JSON function call parser
 */
export interface ParseJsonFunctionCallsOptions {
  /** Support XML-style tags: <tool_call>...</tool_call> */
  supportXmlTags?: boolean;
  /** Support Python-style: [functionName(arg="value")] */
  supportPythonStyle?: boolean;
  /** Support "parameters" as alias for "arguments" (Llama format) */
  supportParametersField?: boolean;
  /** Support call:name{key:value} style delimited with <|tool_call>...<tool_call|> */
  supportCallColonStyle?: boolean;
}

const DEFAULT_OPTIONS: ParseJsonFunctionCallsOptions = {
  supportXmlTags: true,
  supportPythonStyle: true,
  supportParametersField: true,
  supportCallColonStyle: true,
};

function generateToolCallId(): string {
  return `call_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

/**
 * Parses key:value parameter pairs from the call:name{key:value,...} format.
 * Values are coerced to numbers/booleans/null when possible.
 */
function parseCallColonParams(params: string): Record<string, unknown> {
  const args: Record<string, unknown> = {};
  if (!params || !params.trim()) return args;

  const pairs = params.split(",").map((s) => s.trim());
  for (const pair of pairs) {
    const colonIndex = pair.indexOf(":");
    if (colonIndex > 0) {
      const key = pair.substring(0, colonIndex).trim();
      const rawValue = pair.substring(colonIndex + 1).trim();
      if (rawValue === "true") {
        args[key] = true;
      } else if (rawValue === "false") {
        args[key] = false;
      } else if (rawValue === "null") {
        args[key] = null;
      } else {
        const numValue = Number(rawValue);
        args[key] = !isNaN(numValue) && rawValue !== "" ? numValue : rawValue;
      }
    }
  }
  return args;
}

function buildRegex(options: ParseJsonFunctionCallsOptions): RegExp {
  const patterns: string[] = [];

  // Always support markdown fences
  patterns.push("```tool[_-]?call\\s*([\\s\\S]*?)```");

  if (options.supportXmlTags) {
    patterns.push("<tool_call>\\s*([\\s\\S]*?)\\s*</tool_call>");
  }

  if (options.supportPythonStyle) {
    patterns.push("\\[(\\w+)\\(([^)]*)\\)\\]");
  }

  if (options.supportCallColonStyle) {
    patterns.push("<\\|tool_call>\\s*([\\s\\S]*?)\\s*<tool_call\\|>");
  }

  return new RegExp(patterns.join("|"), "gi");
}

/**
 * Parses JSON-formatted tool calls from model response.
 * Supports multiple formats:
 * 1. Single object: {"name": "tool", "arguments": {...}} or {"name": "tool", "parameters": {...}}
 * 2. Array: [{"name": "tool1", ...}, {"name": "tool2", ...}]
 * 3. Newline-separated objects:
 *    {"name": "tool1", "arguments": {...}}
 *    {"name": "tool2", "arguments": {...}}
 *
 * Note: Handles both "arguments" (OpenAI/Mistral format) and "parameters" (Llama format)
 *
 * @param response - The model's response text to parse
 * @param options - Configuration options for parsing
 * @returns Object containing parsed tool calls and remaining text content
 */
export function parseJsonFunctionCalls(
  response: string,
  options: ParseJsonFunctionCallsOptions = DEFAULT_OPTIONS,
): ParsedResponse {
  const mergedOptions = { ...DEFAULT_OPTIONS, ...options };
  const regex = buildRegex(mergedOptions);

  const matches = Array.from(response.matchAll(regex));
  regex.lastIndex = 0;

  if (matches.length === 0) {
    return { toolCalls: [], textContent: response };
  }

  const toolCalls: ParsedToolCall[] = [];
  let textContent = response;

  for (const match of matches) {
    const fullMatch = match[0];
    textContent = textContent.replace(fullMatch, "");

    try {
      // Check for Python-style match: [functionName(args)]
      if (mergedOptions.supportPythonStyle && match[0].startsWith("[")) {
        const pythonMatch = /\[(\w+)\(([^)]*)\)\]/.exec(match[0]);
        if (pythonMatch) {
          const [, funcName, pythonArgs] = pythonMatch;
          const args: Record<string, unknown> = {};

          if (pythonArgs && pythonArgs.trim()) {
            const argPairs = pythonArgs.split(",").map((s) => s.trim());
            for (const pair of argPairs) {
              const equalIndex = pair.indexOf("=");
              if (equalIndex > 0) {
                const key = pair.substring(0, equalIndex).trim();
                let value = pair.substring(equalIndex + 1).trim();
                if (
                  (value.startsWith('"') && value.endsWith('"')) ||
                  (value.startsWith("'") && value.endsWith("'"))
                ) {
                  value = value.substring(1, value.length - 1);
                }
                args[key] = value;
              }
            }
          }

          toolCalls.push({
            type: "tool-call",
            toolCallId: generateToolCallId(),
            toolName: funcName,
            args: args,
          });
          continue;
        }
      }

      // Check for call:name{params} style (inside <|tool_call> delimiters)
      if (mergedOptions.supportCallColonStyle) {
        const callMatch = fullMatch.match(/call:(\w+)\{([^}]*)\}/);
        if (callMatch) {
          const [, funcName, params] = callMatch;
          toolCalls.push({
            type: "tool-call",
            toolCallId: generateToolCallId(),
            toolName: funcName,
            args: parseCallColonParams(params),
          });
          continue;
        }
      }

      // Get the captured content from the first capturing group
      const innerContent = match.slice(1).find((g) => g !== undefined) || "";
      const trimmed = innerContent.trim();

      if (!trimmed) continue;

      // Try parsing as a single JSON value first (object or array)
      try {
        const parsed = JSON.parse(trimmed);
        const callsArray = Array.isArray(parsed) ? parsed : [parsed];

        for (const call of callsArray) {
          if (!call.name) continue;

          let args =
            call.arguments ||
            (mergedOptions.supportParametersField ? call.parameters : null) ||
            {};

          // If args is a string, try to parse it as JSON
          if (typeof args === "string") {
            try {
              args = JSON.parse(args);
            } catch {
              // If parsing fails, keep it as string
            }
          }

          toolCalls.push({
            type: "tool-call",
            toolCallId: call.id || generateToolCallId(),
            toolName: call.name,
            args: args,
          });
        }
      } catch {
        // If single JSON parsing fails, try parsing as newline-separated JSON objects
        const lines = trimmed.split("\n").filter((line) => line.trim());

        for (const line of lines) {
          try {
            const call = JSON.parse(line.trim());
            if (!call.name) continue;

            let args =
              call.arguments ||
              (mergedOptions.supportParametersField ? call.parameters : null) ||
              {};

            if (typeof args === "string") {
              try {
                args = JSON.parse(args);
              } catch {
                // If parsing fails, keep it as string
              }
            }

            toolCalls.push({
              type: "tool-call",
              toolCallId: call.id || generateToolCallId(),
              toolName: call.name,
              args: args,
            });
          } catch {
            // Skip invalid JSON lines
            continue;
          }
        }
      }
    } catch (error) {
      console.warn("Failed to parse JSON tool call:", error);
      continue;
    }
  }

  textContent = textContent.replace(/\n{2,}/g, "\n");

  return { toolCalls, textContent: textContent.trim() };
}

/**
 * Checks if a response contains JSON function calls
 */
export function hasJsonFunctionCalls(
  response: string,
  options: ParseJsonFunctionCallsOptions = DEFAULT_OPTIONS,
): boolean {
  const regex = buildRegex({ ...DEFAULT_OPTIONS, ...options });
  const hasMatch = regex.test(response);
  regex.lastIndex = 0;
  return hasMatch;
}

/**
 * Extracts the first JSON function call block from a response
 */
export function extractJsonFunctionCallsBlock(
  response: string,
  options: ParseJsonFunctionCallsOptions = DEFAULT_OPTIONS,
): string | null {
  const regex = buildRegex({ ...DEFAULT_OPTIONS, ...options });
  const match = regex.exec(response);
  regex.lastIndex = 0;
  return match ? match[0] : null;
}
