import { describe, it, expect } from "vitest";
import { buildApplyChatTemplateOptions } from "../src/utils/generation-helpers";

describe("buildApplyChatTemplateOptions", () => {
  it("passes enable_thinking when set, true or false, and leaves the model's default otherwise", () => {
    expect(
      buildApplyChatTemplateOptions({ enableThinking: true }),
    ).toMatchObject({ enable_thinking: true });
    expect(
      buildApplyChatTemplateOptions({ enableThinking: false }),
    ).toMatchObject({ enable_thinking: false });
    expect(buildApplyChatTemplateOptions({})).not.toHaveProperty(
      "enable_thinking",
    );
  });
});
