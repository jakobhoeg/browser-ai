import { describe, it, expect } from "vitest";
import {
  createUnsupportedSettingWarning,
  createUnsupportedToolWarning,
  gatherUnsupportedSettingWarnings,
} from "../src/utils/warnings";
import type { LanguageModelV4ProviderTool } from "@ai-sdk/provider";

describe("warnings", () => {
  describe("createUnsupportedSettingWarning", () => {
    it("should create a warning with correct structure", () => {
      const warning = createUnsupportedSettingWarning(
        "maxOutputTokens",
        "maxOutputTokens is not supported by Prompt API",
      );

      expect(warning).toEqual({
        type: "unsupported",
        feature: "maxOutputTokens",
        details: "maxOutputTokens is not supported by Prompt API",
      });
    });

    it("should handle different setting names", () => {
      const warning = createUnsupportedSettingWarning(
        "topP",
        "topP is not supported",
      );

      expect(warning).toMatchObject({
        type: "unsupported",
        feature: "topP",
        details: "topP is not supported",
      });
    });
  });

  describe("createUnsupportedToolWarning", () => {
    it("should create a warning with tool information", () => {
      const mockTool: LanguageModelV4ProviderTool = {
        type: "provider",
        id: "test.tool",
        name: "testTool",
        args: {},
      };

      const warning = createUnsupportedToolWarning(
        mockTool,
        "Only function tools are supported by the Prompt API polyfill",
      );

      expect(warning).toEqual({
        type: "unsupported",
        feature: "tool:testTool",
        details: "Only function tools are supported by the Prompt API polyfill",
      });
    });
  });

  describe("gatherUnsupportedSettingWarnings", () => {
    it("should return empty array for empty options", () => {
      const warnings = gatherUnsupportedSettingWarnings({});
      expect(warnings).toEqual([]);
    });

    it("should warn about maxOutputTokens", () => {
      const warnings = gatherUnsupportedSettingWarnings({
        maxOutputTokens: 100,
      });

      expect(warnings).toHaveLength(1);
      expect(warnings[0]).toEqual({
        type: "unsupported",
        feature: "maxOutputTokens",
        details: "maxOutputTokens is not supported by Prompt API",
      });
    });

    it("should warn about stopSequences", () => {
      const warnings = gatherUnsupportedSettingWarnings({
        stopSequences: ["\n"],
      });

      expect(warnings).toHaveLength(1);
      expect(warnings[0]).toEqual({
        type: "unsupported",
        feature: "stopSequences",
        details: "stopSequences is not supported by Prompt API",
      });
    });

    it("should warn about topP", () => {
      const warnings = gatherUnsupportedSettingWarnings({
        topP: 0.9,
      });

      expect(warnings).toHaveLength(1);
      expect(warnings[0]).toEqual({
        type: "unsupported",
        feature: "topP",
        details: "topP is not supported by Prompt API",
      });
    });

    it("should warn about presencePenalty", () => {
      const warnings = gatherUnsupportedSettingWarnings({
        presencePenalty: 0.5,
      });

      expect(warnings).toHaveLength(1);
      expect(warnings[0]).toEqual({
        type: "unsupported",
        feature: "presencePenalty",
        details: "presencePenalty is not supported by Prompt API",
      });
    });

    it("should warn about frequencyPenalty", () => {
      const warnings = gatherUnsupportedSettingWarnings({
        frequencyPenalty: 0.5,
      });

      expect(warnings).toHaveLength(1);
      expect(warnings[0]).toEqual({
        type: "unsupported",
        feature: "frequencyPenalty",
        details: "frequencyPenalty is not supported by Prompt API",
      });
    });

    it("should warn about seed", () => {
      const warnings = gatherUnsupportedSettingWarnings({
        seed: 42,
      });

      expect(warnings).toHaveLength(1);
      expect(warnings[0]).toEqual({
        type: "unsupported",
        feature: "seed",
        details: "seed is not supported by Prompt API",
      });
    });

    it("should warn about toolChoice", () => {
      const warnings = gatherUnsupportedSettingWarnings({
        toolChoice: { type: "auto" },
      });

      expect(warnings).toHaveLength(1);
      expect(warnings[0]).toEqual({
        type: "unsupported",
        feature: "toolChoice",
        details: "toolChoice is not supported by Prompt API",
      });
    });

    it("should gather multiple warnings", () => {
      const warnings = gatherUnsupportedSettingWarnings({
        maxOutputTokens: 100,
        topP: 0.9,
        presencePenalty: 0.5,
        seed: 42,
      });

      expect(warnings).toHaveLength(4);
      expect(
        warnings.filter((w) => w.type === "unsupported").map((w) => w.feature),
      ).toEqual(["maxOutputTokens", "topP", "presencePenalty", "seed"]);
    });

    it("should not warn about undefined options", () => {
      const warnings = gatherUnsupportedSettingWarnings({
        maxOutputTokens: undefined,
        topP: undefined,
      });

      expect(warnings).toEqual([]);
    });

    it("should not warn about supported temperature option", () => {
      const warnings = gatherUnsupportedSettingWarnings({
        temperature: 0.7,
      } as any);

      expect(warnings).toEqual([]);
    });
  });
});
