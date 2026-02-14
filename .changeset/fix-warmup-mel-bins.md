---
"@browser-ai/transformers-js": patch
---

Read num_mel_bins from model config during warmup instead of hardcoding 80. Fixes transcription failures with Whisper large-v3 models (128 mel bins).
