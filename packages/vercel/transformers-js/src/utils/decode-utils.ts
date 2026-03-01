import type {
  PreTrainedTokenizer,
  Processor,
  Tensor,
} from "@huggingface/transformers";

interface TokenSequence {
  data?: number[] | Int32Array | Float32Array;
}

type GenerationOutput =
  | Tensor
  | TokenSequence
  | number[]
  | { sequences?: TokenSequence[] };

/**
 * Decodes generated text from model output sequences, handling both vision and text models
 * @param processor - The tokenizer/processor instance
 * @param sequences - Output sequences from model.generate()
 * @param isVision - Whether this is a vision model
 * @param inputLength - Length of input tokens (only used for text models)
 * @returns Array of decoded text strings
 */
export function decodeGeneratedText(
  processor: PreTrainedTokenizer | Processor,
  sequences: GenerationOutput | GenerationOutput[],
  isVision: boolean,
  inputLength: number,
): string[] {
  if (isVision) {
    return (processor as Processor).batch_decode(sequences as Tensor, {
      skip_special_tokens: true,
    });
  }

  const sequenceArray = Array.isArray(sequences) ? sequences : [sequences];

  return sequenceArray.map((seq: GenerationOutput) => {
    const outputData = (seq as TokenSequence).data || seq;
    const tokenArray = Array.isArray(outputData)
      ? outputData
      : Array.from(outputData as ArrayLike<number>);

    // Extract only new tokens if output includes input
    const newTokens =
      tokenArray.length > inputLength
        ? tokenArray.slice(inputLength)
        : tokenArray;

    return newTokens.length > 0
      ? (processor as PreTrainedTokenizer).decode(newTokens, {
          skip_special_tokens: true,
        })
      : "";
  });
}

/**
 * Decodes a single sequence for text models (used in main thread)
 * @param processor - The tokenizer instance
 * @param sequence - Single output sequence
 * @param inputLength - Length of input tokens
 * @returns Decoded text string
 */
export function decodeSingleSequence(
  processor: PreTrainedTokenizer,
  sequence: GenerationOutput,
  inputLength: number,
): string {
  const outputData = (sequence as TokenSequence).data || sequence;
  const tokenArray = Array.isArray(outputData)
    ? outputData
    : Array.from(outputData as ArrayLike<number>);

  const newTokens =
    tokenArray.length > inputLength
      ? tokenArray.slice(inputLength)
      : tokenArray;

  return newTokens.length > 0
    ? processor.decode(newTokens, { skip_special_tokens: true })
    : "";
}
