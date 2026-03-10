/**
 * Type augmentation for the Prompt API's renamed properties.
 *
 * The @types/dom-chromium-ai package still uses the old property names
 * (inputUsage, inputQuota, measureInputUsage, onquotaoverflow). The spec
 * has renamed these and removed them from non-extension web contexts:
 *
 *   inputUsage         → contextUsage
 *   inputQuota         → contextWindow
 *   measureInputUsage  → measureContextUsage
 *   onquotaoverflow    → oncontextoverflow
 *
 * This file merges the new names into the existing LanguageModel global interface
 * so TypeScript can resolve them. Once @types/dom-chromium-ai ships these names
 * natively this file can be deleted.
 */

declare global {
  interface LanguageModel {
    /** @see https://github.com/webmachinelearning/prompt-api */
    readonly contextUsage: number;
    /** @see https://github.com/webmachinelearning/prompt-api */
    readonly contextWindow: number;
    /** @see https://github.com/webmachinelearning/prompt-api */
    measureContextUsage(
      input: LanguageModelPrompt,
      options?: LanguageModelPromptOptions,
    ): Promise<number>;
    /** @see https://github.com/webmachinelearning/prompt-api */
    oncontextoverflow: ((this: LanguageModel, ev: Event) => unknown) | null;
  }
}

export {};
