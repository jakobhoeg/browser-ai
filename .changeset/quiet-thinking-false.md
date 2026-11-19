---
"@browser-ai/transformers-js": patch
---

Pass an explicit `enableThinking: false` through to the chat template as `enable_thinking: false`

Qwen3's template only turns thinking off on an explicit false. Leaving the option out keeps the model's own default, so nothing changes for users who don't set it.
