/**
 * SessionManager handles the lifecycle of browser AI sessions
 * Manages session creation, caching, availability checks, and progress monitoring
 */

import { LoadSettingError } from "@ai-sdk/provider";
import type { DownloadProgressCallback } from "@browser-ai/shared";

/**
 * Custom provider options that extend the standard API
 */
interface CustomProviderOptions {
  /**
   * Callback invoked when the model context window is exceeded.
   * Replaces the deprecated `onQuotaOverflow`.
   */
  onContextOverflow?: (event: Event) => void;
  /**
   * @deprecated Use `onContextOverflow` instead.
   */
  onQuotaOverflow?: (event: Event) => void;
}

/**
 * Options for creating a new session
 */
export interface SessionCreateOptions
  extends LanguageModelCreateOptions, CustomProviderOptions {
  systemMessage?: string;
  expectedInputs?: Array<{ type: "text" | "image" | "audio" }>;
  onDownloadProgress?: DownloadProgressCallback;
}

/**
 * Manages browser AI session lifecycle
 *
 * Responsibilities:
 * - Create and cache AI sessions
 * - Check model availability
 * - Monitor download progress
 * - Handle session options and configuration
 *
 * @example
 * ```typescript
 * const manager = new SessionManager(config);
 *
 * // Check availability first
 * const status = await manager.checkAvailability();
 *
 * // Create session with progress tracking
 * const session = await manager.getSession({
 *   temperature: 0.7,
 *   onDownloadProgress: (progress) => console.log(`${progress * 100}%`)
 * });
 * ```
 */
export class SessionManager {
  private session: LanguageModel | null = null;
  private baseOptions: LanguageModelCreateOptions &
    Partial<CustomProviderOptions>;

  /**
   * Creates a new SessionManager
   *
   * @param baseOptions - Base configuration options for all sessions
   */
  constructor(
    baseOptions: LanguageModelCreateOptions & Partial<CustomProviderOptions>,
  ) {
    // Filter out our custom options that aren't part of LanguageModelCreateOptions
    this.baseOptions = baseOptions;
  }

  /**
   * Gets or creates a session with the specified options
   *
   * If a session already exists, it will be reused unless force create is needed.
   *
   * @param options - Optional session creation options
   * @returns Promise resolving to a LanguageModel session
   * @throws {LoadSettingError} When Prompt API is not available or model is unavailable
   *
   * @example
   * ```typescript
   * const session = await manager.getSession({
   *   systemMessage: "You are a helpful assistant",
   *   expectedInputs: [{ type: "image" }],
   *   temperature: 0.8
   * });
   * ```
   */
  async getSession(options?: SessionCreateOptions): Promise<LanguageModel> {
    // Check if LanguageModel API is available
    if (typeof LanguageModel === "undefined") {
      throw new LoadSettingError({
        message:
          "Prompt API is not available. This library requires Chrome or Edge browser with browser AI capabilities.",
      });
    }

    // Return existing session if available
    if (this.session) {
      return this.session;
    }

    // Check availability before attempting to create
    const availability = await LanguageModel.availability();
    if (availability === "unavailable") {
      throw new LoadSettingError({
        message: "Built-in model not available in this browser",
      });
    }

    // Prepare session options
    const sessionOptions = this.prepareSessionOptions(options);

    // Create the session
    this.session = await LanguageModel.create(sessionOptions);

    // Resolve overflow handler: onContextOverflow takes precedence, onQuotaOverflow is the deprecated alias
    const onOverflow =
      options?.onContextOverflow ||
      this.baseOptions.onContextOverflow ||
      options?.onQuotaOverflow ||
      this.baseOptions.onQuotaOverflow;

    const overflowHandler =
      onOverflow ??
      (() => {
        console.warn(
          "Model context window exceeded. Consider handling the 'contextoverflow' event.",
        );
      });

    // Register on the new event name; fall back to the old name for older Chrome builds.
    // Cast to `object` in the `in` check to prevent TypeScript narrowing `this.session`
    // to `never` in the else branch (our augmentation declares oncontextoverflow on all
    // LanguageModel instances, but older builds may not have it at runtime).
    const session = this.session;
    if ("oncontextoverflow" in (session as object)) {
      session.addEventListener("contextoverflow", overflowHandler);
    } else {
      session.addEventListener("quotaoverflow", overflowHandler);
    }
    return this.session;
  }

  /**
   * Creates a session with download progress monitoring
   *
   * This is a convenience method for users who want explicit progress tracking.
   *
   * @param onDownloadProgress - Optional callback receiving progress values from 0 to 1
   * @returns Promise resolving to a LanguageModel session
   * @throws {LoadSettingError} When Prompt API is not available or model is unavailable
   *
   * @example
   * ```typescript
   * const session = await manager.createSessionWithProgress(
   *   (progress) => {
   *     console.log(`Download: ${Math.round(progress * 100)}%`);
   *   }
   * );
   * ```
   */
  async createSessionWithProgress(
    onDownloadProgress?: DownloadProgressCallback,
  ): Promise<LanguageModel> {
    return this.getSession({ onDownloadProgress });
  }

  /**
   * Checks the availability status of the browser AI model
   *
   * @returns Promise resolving to availability status
   * - "unavailable": Model is not supported
   * - "downloadable": Model needs to be downloaded
   * - "downloading": Model is currently downloading
   * - "available": Model is ready to use
   *
   * @example
   * ```typescript
   * const status = await manager.checkAvailability();
   * if (status === "downloadable") {
   *   console.log("Model needs to be downloaded first");
   * }
   * ```
   */
  async checkAvailability(): Promise<Availability> {
    if (typeof LanguageModel === "undefined") {
      return "unavailable";
    }
    return LanguageModel.availability();
  }

  /**
   * Gets the current session if it exists
   *
   * @returns The current session or null if none exists
   */
  getCurrentSession(): LanguageModel | null {
    return this.session;
  }

  /**
   * Destroys the current session
   *
   * Use this when you want to force creation of a new session
   * with different options on the next getSession call.
   */
  destroySession(): void {
    if (this.session && typeof this.session.destroy === "function") {
      this.session.destroy();
    }
    this.session = null;
  }

  /**
   * Gets the context window size (token limit) for the current session, if available.
   * @returns The context window size or undefined if not available
   */
  getContextWindow(): number | undefined {
    const session = this.getCurrentSession();
    if (!session) return undefined;
    // contextWindow is the new name; fall back to inputQuota for older Chrome builds
    return (session as LanguageModel).contextWindow ?? session.inputQuota;
  }

  /**
   * Gets the current context usage (tokens consumed) for the current session, if available.
   * @returns The context usage or undefined if not available
   */
  getContextUsage(): number | undefined {
    const session = this.getCurrentSession();
    if (!session) return undefined;
    // contextUsage is the new name; fall back to inputUsage for older Chrome builds
    return (session as LanguageModel).contextUsage ?? session.inputUsage;
  }

  /**
   * @deprecated Use {@link getContextWindow} instead.
   */
  getInputQuota(): number | undefined {
    return this.getContextWindow();
  }

  /**
   * @deprecated Use {@link getContextUsage} instead.
   */
  getInputUsage(): number | undefined {
    return this.getContextUsage();
  }

  /**
   * Prepares merged session options from base config and request options
   *
   * @param options - Optional request-specific options
   * @returns Merged and sanitized options ready for LanguageModel.create()
   * @private
   */
  private prepareSessionOptions(
    options?: SessionCreateOptions,
  ): LanguageModelCreateOptions {
    // Start with base options
    const mergedOptions: LanguageModelCreateOptions &
      Partial<CustomProviderOptions> = { ...this.baseOptions };

    // Merge in request-specific options if provided
    if (options) {
      const {
        systemMessage,
        expectedInputs,
        onDownloadProgress,
        onContextOverflow: _onContextOverflow,
        onQuotaOverflow: _onQuotaOverflow,
        ...createOptions
      } = options;

      // Merge standard create options
      Object.assign(mergedOptions, createOptions);

      // Handle system message
      if (systemMessage) {
        mergedOptions.initialPrompts = [
          { role: "system", content: systemMessage },
        ];
      }

      // Handle expected inputs (for multimodal)
      if (expectedInputs && expectedInputs.length > 0) {
        mergedOptions.expectedInputs = expectedInputs;
      }

      // Handle download progress monitoring
      if (onDownloadProgress) {
        mergedOptions.monitor = (m: CreateMonitor) => {
          m.addEventListener("downloadprogress", (e: ProgressEvent) => {
            onDownloadProgress(e.loaded); // e.loaded is between 0 and 1
          });
        };
      }
    }

    // Remove any custom options that aren't part of the standard API
    this.sanitizeOptions(mergedOptions);

    return mergedOptions;
  }

  /**
   * Removes custom options that aren't part of LanguageModel.create API
   *
   * @param options - Options object to sanitize in-place
   * @private
   */
  private sanitizeOptions(
    options: LanguageModelCreateOptions & Partial<CustomProviderOptions>,
  ): void {
    // Remove our custom options that the Prompt API doesn't understand
  }
}
