import { openai } from "@ai-sdk/openai";
import {
  convertToModelMessages,
  createUIMessageStreamResponse,
  streamText,
  toUIMessageStream,
  UIMessage,
} from "ai";
import z from "zod";

export const maxDuration = 30;

export async function POST(req: Request) {
  const { messages }: { messages: UIMessage[] } = await req.json();

  const prompt = await convertToModelMessages(messages);

  const result = streamText({
    model: openai("gpt-4o"),
    messages: prompt,
    tools: {
      getWeatherInformation: {
        description: "Get the weather for a location",
        inputSchema: z.object({
          location: z.string().describe("City and country, e.g. Paris, FR"),
          format: z
            .enum(["celsius", "fahrenheit"])
            .describe("Temperature unit"),
        }),
        execute: async ({ location, format }) => {
          console.log("TOOL EXECUTED");
          return `Mock weather in ${location}: 25°${format === "celsius" ? "C" : "F"} and sunny.`;
        },
      },
    },
  });

  return createUIMessageStreamResponse({
    stream: toUIMessageStream({ stream: result.stream }),
  });
}
