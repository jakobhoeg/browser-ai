import path from "path";
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["test/setup.ts"],
  },
  resolve: {
    alias: {
      "@browser-ai/shared/testing": path.resolve(
        __dirname,
        "../shared/src/testing.ts",
      ),
      "@browser-ai/shared": path.resolve(__dirname, "../shared/src/index.ts"),
    },
  },
});
