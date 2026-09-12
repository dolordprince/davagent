/**
 * davagent — Cloudflare Worker
 * Exposes OpenAI-compatible endpoints backed by Workers AI (free tier)
 * Callable from: curl, python openai SDK, OpenHands SDK, any terminal
 *
 * Routes:
 *   GET  /              — health check + route list
 *   GET  /v1/models     — list available models (OpenAI format)
 *   POST /v1/chat/completions  — chat endpoint (streaming + non-streaming)
 *   POST /v1/completions       — legacy completions endpoint
 *   POST /agent/run            — OpenHands-style task dispatch
 */

export interface Env {
  AI: Ai;
  AGENT_SECRET?: string;
}

const MODELS: Record<string, string> = {
  "llama-4-scout":   "@cf/meta/llama-4-scout-17b-16e-instruct",
  "llama-3.3-70b":   "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
  "llama-3.1-8b":    "@cf/meta/llama-3.1-8b-instruct",
  "glm-4.7":         "@cf/zai-org/glm-4.7-flash",
  "mistral-7b":      "@cf/mistralai/mistral-7b-instruct-v0.1",
  "qwen-14b":        "@cf/qwen/qwen1.5-14b-chat-awq",
  "deepseek-r1":     "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
  "default":         "@cf/meta/llama-4-scout-17b-16e-instruct",
};

const DEFAULT_MODEL = "llama-4-scout";

const SYSTEM_PROMPT = `You are DAV AGENT — an expert software engineering assistant built by Dolor David Prince (dolordprince on GitHub). You specialize in:
- Full-stack development (FastAPI, React, Next.js, TypeScript)
- Python agent systems and MCP tools
- Cloudflare Workers, HuggingFace Spaces, Render deployments
- Android/Termux development workflows
- Writing production-ready code with no mocks, no placeholders

When writing code: always complete, always working, no TODOs. When asked for scripts: use heredoc format compatible with bash.`;

function corsHeaders(): HeadersInit {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
  };
}

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { ...corsHeaders(), "Content-Type": "application/json" },
  });
}

function err(message: string, status = 400): Response {
  return json({ error: { message, type: "invalid_request_error" } }, status);
}

function resolveModel(requested?: string): string {
  if (!requested) return MODELS[DEFAULT_MODEL];
  if (requested.startsWith("@cf/")) return requested;
  return MODELS[requested] ?? MODELS[DEFAULT_MODEL];
}

// Normalize CF result → plain string across all model families:
//   { response: "..." }                              most models
//   { choices: [{ message: { content: "..." } }] }  GLM-4.7 / OpenAI-compat
//   { choices: [{ text: "..." }] }                  legacy shape
function extractText(result: any): string {
  if (typeof result?.response === "string" && result.response !== "")
    return result.response;
  const msg = result?.choices?.[0]?.message;
  // content field (standard)
  if (typeof msg?.content === "string" && msg.content !== "")
    return msg.content;
  // reasoning field fallback (GLM-4.7 / DeepSeek when content is null)
  if (typeof msg?.reasoning === "string" && msg.reasoning !== "")
    return msg.reasoning;
  if (typeof msg?.reasoning_content === "string" && msg.reasoning_content !== "")
    return msg.reasoning_content;
  if (typeof result?.choices?.[0]?.text === "string" && result.choices[0].text !== "")
    return result.choices[0].text;
  return JSON.stringify(result ?? "");
}

// GLM-4.7 and DeepSeek-R1 emit delta.reasoning before delta.content.
// Strip reasoning-only chunks so clients receive only the final answer tokens.
function makeReasoningStripStream(
  source: ReadableStream,
  requestId: string,
  modelAlias: string,
): ReadableStream {
  const { readable, writable } = new TransformStream();
  const writer = writable.getWriter();
  const encoder = new TextEncoder();

  (async () => {
    const reader = source.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          if (trimmed === "data: [DONE]") {
            await writer.write(encoder.encode("data: [DONE]\n\n"));
            continue;
          }
          if (!trimmed.startsWith("data:")) continue;
          let chunk: any;
          try { chunk = JSON.parse(trimmed.slice(5).trim()); } catch { continue; }
          const delta = chunk?.choices?.[0]?.delta ?? {};
          // Skip chunks that only carry reasoning tokens (no content key at all)
          if (!("content" in delta)) continue;
          const content: string = typeof delta.content === "string" ? delta.content : "";
          const out = {
            id: requestId,
            object: "chat.completion.chunk",
            created: chunk.created ?? Math.floor(Date.now() / 1000),
            model: modelAlias,
            choices: [{
              index: 0,
              delta: { content },
              finish_reason: chunk?.choices?.[0]?.finish_reason ?? null,
            }],
          };
          await writer.write(encoder.encode(`data: ${JSON.stringify(out)}\n\n`));
        }
      }
      await writer.write(encoder.encode("data: [DONE]\n\n"));
    } finally {
      writer.close();
    }
  })();

  return readable;
}

function handleModels(): Response {
  const models = Object.keys(MODELS)
    .filter((k) => k !== "default")
    .map((id) => ({
      id,
      object: "model",
      created: 1700000000,
      owned_by: "cloudflare-workers-ai",
      cf_model_id: MODELS[id],
    }));
  return json({ object: "list", data: models });
}

async function handleChatCompletions(request: Request, env: Env): Promise<Response> {
  let body: any;
  try { body = await request.json(); } catch { return err("Invalid JSON body"); }

  const messages = body.messages as { role: string; content: string }[];
  if (!messages?.length) return err("messages array is required");

  const modelId = resolveModel(body.model);
  const modelAlias: string = body.model ?? DEFAULT_MODEL;
  const stream: boolean = body.stream === true;
  const maxTokens: number = body.max_tokens ?? 4096;

  const hasSystem = messages.some((m) => m.role === "system");
  const finalMessages = hasSystem
    ? messages
    : [{ role: "system", content: SYSTEM_PROMPT }, ...messages];

  if (stream) {
    const aiStream = await env.AI.run(modelId as any, {
      messages: finalMessages,
      stream: true,
      max_tokens: maxTokens,
    } as any);

    const requestId = `chatcmpl-${Date.now()}`;
    const outStream = makeReasoningStripStream(
      aiStream as ReadableStream,
      requestId,
      modelAlias,
    );

    return new Response(outStream, {
      headers: {
        ...corsHeaders(),
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
      },
    });
  }

  const result = await env.AI.run(modelId as any, {
    messages: finalMessages,
    max_tokens: maxTokens,
  } as any) as any;

  return json({
    id: `chatcmpl-${Date.now()}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: modelAlias,
    choices: [{
      index: 0,
      message: { role: "assistant", content: extractText(result) },
      finish_reason: "stop",
    }],
    usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
  });
}

async function handleCompletions(request: Request, env: Env): Promise<Response> {
  let body: any;
  try { body = await request.json(); } catch { return err("Invalid JSON body"); }

  const prompt: string = body.prompt ?? "";
  if (!prompt) return err("prompt is required");

  const modelId = resolveModel(body.model);
  const result = await env.AI.run(modelId as any, {
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: prompt },
    ],
    max_tokens: body.max_tokens ?? 2048,
  } as any) as any;

  return json({
    id: `cmpl-${Date.now()}`,
    object: "text_completion",
    created: Math.floor(Date.now() / 1000),
    model: body.model ?? DEFAULT_MODEL,
    choices: [{ text: extractText(result), index: 0, finish_reason: "stop" }],
  });
}

async function handleAgentRun(request: Request, env: Env): Promise<Response> {
  if (env.AGENT_SECRET) {
    const auth = request.headers.get("Authorization") ?? "";
    const token = auth.replace("Bearer ", "").trim();
    if (token !== env.AGENT_SECRET) return err("Unauthorized", 401);
  }

  let body: any;
  try { body = await request.json(); } catch { return err("Invalid JSON body"); }

  const task: string = body.task ?? body.message ?? "";
  if (!task) return err("task or message field is required");

  const modelId = resolveModel(body.model);
  const context: string = body.context ?? "";

  const systemMsg = `${SYSTEM_PROMPT}

You are operating as an autonomous agent. The user will give you a task. Think step by step, produce a complete solution, and format your response as:

PLAN:
<numbered steps>

IMPLEMENTATION:
<complete code or commands>

RESULT:
<summary of what was done>`;

  const userMsg = context
    ? `Context:\n${context}\n\nTask:\n${task}`
    : `Task:\n${task}`;

  const result = await env.AI.run(modelId as any, {
    messages: [
      { role: "system", content: systemMsg },
      { role: "user", content: userMsg },
    ],
    max_tokens: body.max_tokens ?? 4096,
  } as any) as any;

  return json({
    id: `agent-${Date.now()}`,
    task,
    model: body.model ?? DEFAULT_MODEL,
    response: extractText(result),
    status: "complete",
  });
}

function handleRoot(): Response {
  return json({
    name: "davagent",
    version: "1.0.0",
    author: "Dolor David Prince",
    status: "online",
    endpoints: {
      "GET  /":                    "this health check",
      "GET  /v1/models":           "list available models",
      "POST /v1/chat/completions": "OpenAI-compatible chat (streaming supported)",
      "POST /v1/completions":      "OpenAI-compatible legacy completions",
      "POST /agent/run":           "OpenHands-style task dispatch",
    },
  });
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const method = request.method.toUpperCase();

    if (method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders() });
    }

    const path = url.pathname;

    if (method === "GET" && (path === "/" || path === "")) return handleRoot();
    if (method === "GET" && path === "/v1/models") return handleModels();
    if (method === "POST" && path === "/v1/chat/completions") return handleChatCompletions(request, env);
    if (method === "POST" && path === "/v1/completions") return handleCompletions(request, env);
    if (method === "POST" && path === "/agent/run") return handleAgentRun(request, env);

    return err(`Route not found: ${method} ${path}`, 404);
  },
} satisfies ExportedHandler<Env>;
