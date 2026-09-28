import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm", "cjs"],
  dts: true,
  splitting: false,
  sourcemap: true,
  clean: true,
  // Externalize the heavy runtimes; bundle our shared utilities.
  external: ["@litert-lm/core", "ai"],
  noExternal: ["@browser-ai/shared"],
});
