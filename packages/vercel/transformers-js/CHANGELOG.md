# @browser-ai/transformers-js

## 2.1.8

### Patch Changes

- 3649282: chore: add jsDelivr to readme

## 2.1.7

### Patch Changes

- f2ff3ba: feat: allow using @huggingface/transformers.js@4.0.0-next

## 2.1.6

### Patch Changes

- a34bcea: refactor: extract shared streaming processor to eliminate duplicated doStream logic

## 2.1.5

### Patch Changes

- 254330a: fix: guard transcription warmup when model config is missing

## 2.1.4

### Patch Changes

- c309a09: perf: improve stream parsing perf from O(n²) to O(n)

## 2.1.3

### Patch Changes

- bbe98df: chore: update dependencies
- 4e434b3: fix: add past_key_values cache reuse for worker chat generation

## 2.1.2

### Patch Changes

- e8000c4: fix: worker model loading by preserving use_external_data_format defaults

## 2.1.1

### Patch Changes

- 7486d29: Read num_mel_bins from model config during warmup instead of hardcoding 80. Fixes transcription failures with Whisper large-v3 models (128 mel bins).

## 2.1.0

### Minor Changes

- acc8791: refactor: unify `createSessionWithProgress` to use a `(progress: number) => void` callback across all packages

## 2.0.3

### Patch Changes

- 0f51e16: refactor: extract shared utilities into internal @browser-ai/shared package
- ce8bcb7: allow configuration of local model path and cache dir

## 2.0.2

### Patch Changes

- f8b6996: fix: correct ESM export paths to use .mjs extension

## 2.0.1

### Patch Changes

- 3f665ca: chore: add hero image to npm readme

## 2.0.0

### Major Changes

- 0266287: feat: move package to @browser-ai org
