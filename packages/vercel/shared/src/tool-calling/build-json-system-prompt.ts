import type {
  JSONSchema7,
  LanguageModelV3FunctionTool,
} from "@ai-sdk/provider";
import type { ToolDefinition } from "../types";

/**
 * Builds an enhanced system prompt for JSON-based tool calling.
 * The model receives JSON schemas and is expected to return JSON tool calls.
 *
 * @param originalSystemPrompt - The original system prompt (if any)
 * @param tools - Array of available tool definitions
 * @param options - Configuration options for tool calling behavior (unused, kept for backwards compatibility)
 * @returns Enhanced system prompt with JSON tool calling instructions
 */
export function buildJsonToolSystemPrompt(
  originalSystemPrompt: string | undefined,
  tools: Array<ToolDefinition | LanguageModelV3FunctionTool>,
  options?: { allowParallelToolCalls?: boolean },
): string {
  if (!tools || tools.length === 0) {
    return originalSystemPrompt || "";
  }

  const parallelInstruction =
    "Only request one tool call at a time. Wait for tool results before asking for another tool.";

  const toolSchemas = tools.map((tool) => {
    const schema = getParameters(tool);
    return {
      name: tool.name,
      description: tool.description ?? "No description provided.",
      parameters: schema || { type: "object", properties: {} },
    };
  });

  const toolsJson = JSON.stringify(toolSchemas, null, 2);

  const instructionBody = `You are a helpful AI assistant with access to tools.

# Available Tools
${toolsJson}

# Tool Calling Instructions
${parallelInstruction}

To call a tool, output JSON in this exact format inside a \`\`\`tool_call code fence:

\`\`\`tool_call
{"name": "tool_name", "arguments": {"param1": "value1", "param2": "value2"}}
\`\`\`

Tool responses will be provided in \`\`\`tool_result fences. Each line contains JSON like:
\`\`\`tool_result
{"id": "call_123", "name": "tool_name", "result": {...}, "error": false}
\`\`\`
Use the \`result\` payload (and treat \`error\` as a boolean flag) when continuing the conversation.

Important:
- Use exact tool and parameter names from the schema above
- Arguments must be a valid JSON object matching the tool's parameters
- You can include brief reasoning before or after the tool call
- If no tool is needed, respond directly without tool_call fences`;

  if (originalSystemPrompt?.trim()) {
    return `${originalSystemPrompt.trim()}\n\n${instructionBody}`;
  }

  return instructionBody;
}

/**
 * Extracts the parameters/input schema from a tool definition.
 * Handles both ToolDefinition (parameters field) and LanguageModelV3FunctionTool (inputSchema field).
 *
 * @param tool - The tool definition to extract parameters from
 * @returns The JSON Schema for the tool's parameters, or undefined if not present
 */
function getParameters(
  tool: ToolDefinition | LanguageModelV3FunctionTool,
): JSONSchema7 | undefined {
  if ("parameters" in tool) {
    return tool.parameters;
  }

  return tool.inputSchema as JSONSchema7 | undefined;
}
