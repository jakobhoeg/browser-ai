---
"@browser-ai/wllama": patch
---

Add `@browser-ai/wllama`, a chat provider that runs GGUF models in the browser with llama.cpp's WebGPU backend via [wllama](https://github.com/ngxson/wllama). Supports model loading from Hugging Face or direct URLs with download progress, `generateText` / `streamText`, native tool calling, and JSON schema structured output. Call `unload()` to free the model's memory. Requires `@wllama/wllama` 3.6.1 exactly, matching the WASM binary the provider loads.
