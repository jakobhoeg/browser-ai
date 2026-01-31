/**
 * Warning generation utilities for unsupported settings and tools
 */

import type { SharedV3Warning } from "@ai-sdk/provider";
import { createUnsupportedSettingWarning } from "@browser-ai/shared";

// Re-export shared utilities
export {
  createUnsupportedSettingWarning,
  createUnsupportedToolWarning,
} from "@browser-ai/shared";

/**
 * Gathers all warnings for unsupported call options (Prompt API specific)
 *
 * @param options - The call options to check
 * @returns Array of warnings for any unsupported settings
 *
 * @example
 * ```typescript
 * const warnings = gatherUnsupportedSettingWarnings({
 *   maxOutputTokens: 100,
 *   topP: 0.9,
 *   temperature: 0.7,
 * });
 * // Returns warnings for maxOutputTokens and topP
 * ```
 */
export function gatherUnsupportedSettingWarnings(options: {
  maxOutputTokens?: number;
  stopSequences?: string[];
  topP?: number;
  presencePenalty?: number;
  frequencyPenalty?: number;
  seed?: number;
  toolChoice?: unknown;
}): SharedV3Warning[] {
  const warnings: SharedV3Warning[] = [];

  if (options.maxOutputTokens != null) {
    warnings.push(
      createUnsupportedSettingWarning(
        "maxOutputTokens",
        "maxOutputTokens is not supported by Prompt API",
      ),
    );
  }

  if (options.stopSequences != null) {
    warnings.push(
      createUnsupportedSettingWarning(
        "stopSequences",
        "stopSequences is not supported by Prompt API",
      ),
    );
  }

  if (options.topP != null) {
    warnings.push(
      createUnsupportedSettingWarning(
        "topP",
        "topP is not supported by Prompt API",
      ),
    );
  }

  if (options.presencePenalty != null) {
    warnings.push(
      createUnsupportedSettingWarning(
        "presencePenalty",
        "presencePenalty is not supported by Prompt API",
      ),
    );
  }

  if (options.frequencyPenalty != null) {
    warnings.push(
      createUnsupportedSettingWarning(
        "frequencyPenalty",
        "frequencyPenalty is not supported by Prompt API",
      ),
    );
  }

  if (options.seed != null) {
    warnings.push(
      createUnsupportedSettingWarning(
        "seed",
        "seed is not supported by Prompt API",
      ),
    );
  }

  if (options.toolChoice != null) {
    warnings.push(
      createUnsupportedSettingWarning(
        "toolChoice",
        "toolChoice is not supported by Prompt API",
      ),
    );
  }

  return warnings;
}
