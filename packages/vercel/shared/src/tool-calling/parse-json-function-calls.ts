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
  /** Support <|tool_call_start|>...<|tool_call_end|> delimiters (LFM2 / LFM2.5) */
  supportToolCallStartEnd?: boolean;
}

const DEFAULT_OPTIONS: ParseJsonFunctionCallsOptions = {
  supportXmlTags: true,
  supportPythonStyle: true,
  supportParametersField: true,
  supportCallColonStyle: true,
  supportToolCallStartEnd: true,
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

/**
 * Decides whether the quote at `index` actually terminates the string.
 *
 * Models rarely escape apostrophes, so `'What's the budget?'` is common. A
 * single quote therefore only closes when the next meaningful character is
 * structural; otherwise it is a literal apostrophe. Double quotes are
 * unambiguous and always close.
 */
function closesQuote(text: string, index: number, quote: string): boolean {
  if (quote !== "'") return true;

  for (let i = index + 1; i < text.length; i++) {
    const char = text[i];
    if (char === " " || char === "\t" || char === "\n" || char === "\r") {
      continue;
    }
    return (
      char === "," ||
      char === ":" ||
      char === "}" ||
      char === "]" ||
      char === ")"
    );
  }

  return true; // end of input
}

/**
 * Splits an argument list on commas that are not inside quotes or brackets,
 * so values like `query="Doe, Jane"` survive intact.
 */
function splitArguments(args: string): string[] {
  const parts: string[] = [];
  let current = "";
  let quote: string | null = null;
  let depth = 0;

  for (let i = 0; i < args.length; i++) {
    const char = args[i];

    if (quote) {
      if (char === "\\" && i + 1 < args.length) {
        current += char + args[++i];
        continue;
      }
      if (char === quote && closesQuote(args, i, quote)) quote = null;
      current += char;
      continue;
    }

    if (char === '"' || char === "'") {
      quote = char;
      current += char;
    } else if (char === "[" || char === "{" || char === "(") {
      depth++;
      current += char;
    } else if (char === "]" || char === "}" || char === ")") {
      depth--;
      current += char;
    } else if (char === "," && depth === 0) {
      parts.push(current);
      current = "";
    } else {
      current += char;
    }
  }

  if (current.trim()) parts.push(current);
  return parts.map((part) => part.trim()).filter(Boolean);
}

/**
 * Rewrites a Python literal into JSON: single-quoted strings become
 * double-quoted, and `True`/`False`/`None` become their JSON equivalents.
 * Tokenizes rather than string-replacing so quotes and keywords appearing
 * inside string values are left alone.
 */
function pythonLiteralToJson(input: string): string {
  let out = "";

  for (let i = 0; i < input.length; i++) {
    const char = input[i];

    if (char === "'" || char === '"') {
      const quote = char;
      let value = "";
      i++;

      for (
        ;
        i < input.length &&
        !(input[i] === quote && closesQuote(input, i, quote));
        i++
      ) {
        if (input[i] === "\\" && i + 1 < input.length) {
          // Unescape into the raw value; JSON.stringify re-escapes below
          const next = input[++i];
          value +=
            next === "n"
              ? "\n"
              : next === "t"
                ? "\t"
                : next === "r"
                  ? "\r"
                  : next;
          continue;
        }
        value += input[i];
      }

      out += JSON.stringify(value);
      continue;
    }

    if (/[A-Za-z]/.test(char)) {
      let word = "";
      while (i < input.length && /[A-Za-z]/.test(input[i])) word += input[i++];
      i--;

      out +=
        word === "True"
          ? "true"
          : word === "False"
            ? "false"
            : word === "None"
              ? "null"
              : word;
      continue;
    }

    out += char;
  }

  return out;
}

/**
 * Parses a single Python-style argument value into its JavaScript equivalent.
 * Handles JSON, Python literals (lists, dicts, `True`/`None`), numbers, and
 * quoted strings; anything unrecognized is returned as a trimmed string.
 */
function parsePythonValue(raw: string): unknown {
  const value = raw.trim();
  if (!value) return "";

  const isQuoted =
    (value.startsWith('"') && value.endsWith('"')) ||
    (value.startsWith("'") && value.endsWith("'"));

  // Structured or scalar literals — try JSON first, then Python syntax
  if (!isQuoted) {
    try {
      return JSON.parse(value);
    } catch {
      // not JSON, fall through
    }

    try {
      return JSON.parse(pythonLiteralToJson(value));
    } catch {
      return value;
    }
  }

  // Quoted string: reuse the same tokenizer so escapes are handled
  try {
    return JSON.parse(pythonLiteralToJson(value));
  } catch {
    return value.substring(1, value.length - 1);
  }
}

/**
 * Finds `name(...)` calls in text, tracking quotes and nesting so arguments
 * containing parentheses (e.g. `q="budget (roughly)?"`) are not truncated.
 */
function scanPythonCalls(text: string): Array<{ name: string; args: string }> {
  const calls: Array<{ name: string; args: string }> = [];
  const nameRegex = /(\w+)\s*\(/g;
  let match: RegExpExecArray | null;

  while ((match = nameRegex.exec(text)) !== null) {
    const start = match.index + match[0].length;
    let depth = 1;
    let quote: string | null = null;
    let i = start;

    for (; i < text.length && depth > 0; i++) {
      const char = text[i];

      if (quote) {
        if (char === "\\") i++;
        else if (char === quote && closesQuote(text, i, quote)) quote = null;
        continue;
      }

      if (char === '"' || char === "'") quote = char;
      else if (char === "(") depth++;
      else if (char === ")") depth--;
    }

    // Unbalanced — no closing paren, so this is not a complete call
    if (depth !== 0) continue;

    calls.push({ name: match[1], args: text.slice(start, i - 1) });
    nameRegex.lastIndex = i;
  }

  return calls;
}

/**
 * Parses Python-style calls such as `func(arg="value")`. Accepts several calls
 * separated by commas, with or without the surrounding brackets.
 */
function parsePythonStyleCalls(text: string): ParsedToolCall[] {
  const calls: ParsedToolCall[] = [];

  for (const { name, args: rawArgs } of scanPythonCalls(text)) {
    const args: Record<string, unknown> = {};

    for (const pair of splitArguments(rawArgs)) {
      const equalIndex = pair.indexOf("=");
      if (equalIndex <= 0) continue;

      const key = pair.substring(0, equalIndex).trim();
      args[key] = parsePythonValue(pair.substring(equalIndex + 1));
    }

    calls.push({
      type: "tool-call",
      toolCallId: generateToolCallId(),
      toolName: name,
      args,
    });
  }

  return calls;
}

/**
 * A quoted string, either single or double, with escapes. An unescaped `'` is
 * treated as content unless followed by a structural character, mirroring
 * `closesQuote` so detection agrees with tokenization.
 */
const QUOTED = `"(?:[^"\\\\]|\\\\.)*"|'(?:[^'\\\\]|\\\\.|'(?!\\s*(?:[,:}\\])]|$)))*'`;
/** Argument list body: bare chars, quoted strings, or one level of nesting */
const CALL_ARGS = `(?:[^()"']|${QUOTED}|\\((?:[^()"']|${QUOTED})*\\))*`;
/** A single Python-style call: `name(args)` */
const PY_CALL = `\\w+\\(${CALL_ARGS}\\)`;

function buildRegex(options: ParseJsonFunctionCallsOptions): RegExp {
  const patterns: string[] = [];

  // Always support markdown fences
  patterns.push("```tool[_-]?call\\s*([\\s\\S]*?)```");

  if (options.supportXmlTags) {
    patterns.push("<tool_call>\\s*([\\s\\S]*?)\\s*</tool_call>");
  }

  if (options.supportPythonStyle) {
    // One or more `func(args)` calls inside brackets: [f(a="b"), g(c="d")]
    patterns.push(`\\[\\s*${PY_CALL}(?:\\s*,\\s*${PY_CALL})*\\s*\\]`);
  }

  if (options.supportCallColonStyle) {
    patterns.push("<\\|tool_call>\\s*([\\s\\S]*?)\\s*<tool_call\\|>");
  }

  if (options.supportToolCallStartEnd) {
    patterns.push(
      "<\\|tool_call_start\\|>\\s*([\\s\\S]*?)\\s*<\\|tool_call_end\\|>",
    );
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
      // Check for Python-style match: [functionName(args), ...]
      if (mergedOptions.supportPythonStyle && fullMatch.startsWith("[")) {
        const pythonCalls = parsePythonStyleCalls(fullMatch);
        if (pythonCalls.length > 0) {
          toolCalls.push(...pythonCalls);
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

      // Delimited blocks may wrap Python-style calls rather than JSON
      // (LFM2/LFM2.5 emit `<|tool_call_start|>[f(a="b")]<|tool_call_end|>`)
      if (mergedOptions.supportPythonStyle && /^\[?\s*\w+\(/.test(trimmed)) {
        const pythonCalls = parsePythonStyleCalls(trimmed);
        if (pythonCalls.length > 0) {
          toolCalls.push(...pythonCalls);
          continue;
        }
      }

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
