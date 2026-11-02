// Test setup for @browser-ai/wllama
import { beforeAll } from "vitest";

beforeAll(() => {
  // Mock WebGPU + JSPI for the testing environment
  if (!global.navigator) {
    (global as any).navigator = {};
  }
  if (!(global.navigator as any).gpu) {
    (global.navigator as any).gpu = {};
  }
  if (!(globalThis.WebAssembly as any).Suspending) {
    (globalThis.WebAssembly as any).Suspending = function Suspending() {};
  }
});
