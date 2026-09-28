// Test setup for @browser-ai/litert-lm
import { beforeAll } from "vitest";

beforeAll(() => {
  // Mock WebGPU for the testing environment.
  if (!global.navigator) {
    (global as any).navigator = {} as Navigator;
  }
  if (!(global.navigator as any).gpu) {
    (global.navigator as any).gpu = {};
  }
});
