/**
 * Warning generation utilities for unsupported settings and tools
 */

import type {
  SharedV4Warning,
  LanguageModelV4ProviderTool,
} from "@ai-sdk/provider";

/**
 * Creates a warning for an unsupported setting
 *
 * @param setting - Name of the setting that is not supported
 * @param details - Additional details about why it's not supported
 * @returns A call warning object
 *
 * @example
 * ```typescript
 * const warning = createUnsupportedSettingWarning(
 *   "maxOutputTokens",
 *   "maxOutputTokens is not supported by this provider"
 * );
 * ```
 */
export function createUnsupportedSettingWarning(
  feature: string,
  details: string,
): SharedV4Warning {
  return {
    type: "unsupported",
    feature,
    details,
  };
}

/**
 * Whether a tool choice is the AI SDK's implicit default.
 *
 * The AI SDK normalizes an unset `toolChoice` to `{ type: "auto" }` and passes
 * it to the provider on every call, with or without tools. Auto is what these
 * providers already do, so warning about it would fire on every single
 * request. An explicit `none` / `required` / `tool` choice is a real
 * instruction they cannot honour, and still warrants a warning.
 *
 * @param toolChoice - The tool choice from the call options
 * @returns Whether the choice merely restates the default
 *
 * @example
 * ```typescript
 * isAutoToolChoice({ type: "auto" }); // true
 * isAutoToolChoice({ type: "required" }); // false
 * ```
 */
export function isAutoToolChoice(toolChoice: unknown): boolean {
  return (
    typeof toolChoice === "object" &&
    toolChoice !== null &&
    (toolChoice as { type?: unknown }).type === "auto"
  );
}

/**
 * Creates a warning for an unsupported tool type
 *
 * @param tool - The provider-defined tool that is not supported
 * @param details - Additional details about why it's not supported
 * @returns A call warning object
 *
 * @example
 * ```typescript
 * const warning = createUnsupportedToolWarning(
 *   providerTool,
 *   "Only function tools are supported"
 * );
 * ```
 */
export function createUnsupportedToolWarning(
  tool: LanguageModelV4ProviderTool,
  details: string,
): SharedV4Warning {
  return {
    type: "unsupported",
    feature: `tool:${tool.name}`,
    details,
  };
}
