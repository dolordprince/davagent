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
  // Shape 1: { response: "..." }  — most CF models
  if (typeof result?.response === "string" && result.response !== "")
    return result.response;
  // Shape 2: { choices: [{ message: { content: "..." } }] }  — GLM-4.7 / OpenAI-compat
  // content may be null when model ran out of tokens during reasoning phase;
  // in that case we return "" so caller gets an empty string, not the reasoning chain.
  const msg = result?.choices?.[0]?.message;
  if (msg !== undefined && msg !== null) {
    if (typeof msg.content === "string" && msg.content !== "")
      return msg.content;
    // content is null/empty — model hit token limit in reasoning; return empty
    return "";
  }
  // Shape 3: legacy { choices: [{ text: "..." }] }
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
  const maxTokens: number = Math.max(body.max_tokens ?? 4096, 2048);

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


async function handleResponses(request: Request, env: Env): Promise<Response> {
  let body: any;
  try { body = await request.json(); } catch { return err("Invalid JSON body"); }

  const modelId    = resolveModel(body.model);
  const modelAlias = body.model ?? DEFAULT_MODEL;
  const maxTokens  = body.max_output_tokens ?? body.max_tokens ?? 4096;
  const requestId  = `resp_${Date.now()}`;
  const itemId     = `msg_${Date.now()}`;

  // ── Normalise Responses API input → Workers AI messages ──────────────────
  const messages: { role: string; content: string }[] = [];

  const inputs: any[] = typeof body.input === "string"
    ? [{ type: "message", role: "user", content: body.input }]
    : Array.isArray(body.input) ? body.input
    : Array.isArray(body.messages) ? body.messages
    : [];

  for (const item of inputs) {
    // plain chat message
    if (item.role && (item.content !== undefined || item.type === "message")) {
      const role    = item.role === "developer" ? "system" : (item.role ?? "user");
      const content = typeof item.content === "string" ? item.content
        : Array.isArray(item.content) ? item.content.map((c: any) => c.text ?? c.output ?? "").join("") : "";
      messages.push({ role, content });
      continue;
    }
    // function_call_output  (tool result coming back from Codex)
    if (item.type === "function_call_output") {
      messages.push({ role: "tool", content: typeof item.output === "string" ? item.output : JSON.stringify(item.output ?? "") });
      continue;
    }
    // function_call  (assistant turn in history)
    if (item.type === "function_call") {
      messages.push({ role: "assistant", content: `[tool_call] ${item.name}(${item.arguments ?? ""})` });
      continue;
    }
  }

  if (!messages.some(m => m.role === "system")) {
    messages.unshift({ role: "system", content: SYSTEM_PROMPT });
  }

  // ── Build Workers AI payload ──────────────────────────────────────────────
  const cfPayload: any = { messages, max_tokens: maxTokens };

  if (Array.isArray(body.tools) && body.tools.length > 0 && body.tool_choice !== "none") {
    cfPayload.tools = body.tools.map((t: any) => ({
      type: "function",
      function: {
        name:        t.name ?? t.function?.name,
        description: t.description ?? t.function?.description ?? "",
        parameters:  t.parameters ?? t.function?.parameters ?? { type: "object", properties: {} },
      }
    }));
  }

  // ── Run inference (always non-streaming when tools present) ──────────────
  const result = await env.AI.run(modelId as any, { ...cfPayload, stream: false } as any) as any;

  const msg = result?.choices?.[0]?.message ?? result;

  // Check for tool calls
  const rawCalls: any[] = Array.isArray(msg?.tool_calls)   ? msg.tool_calls
    : Array.isArray(result?.tool_calls) ? result.tool_calls : [];

  if (rawCalls.length > 0) {
    const output = rawCalls.map((tc: any, i: number) => {
      const fn   = tc.function ?? tc;
      const args = typeof fn.arguments === "string" ? fn.arguments : JSON.stringify(fn.arguments ?? {});
      return {
        type:      "function_call",
        id:        `fc_${Date.now()}_${i}`,
        call_id:   tc.id ?? `call_${Date.now()}_${i}`,
        name:      fn.name ?? tc.name ?? "unknown",
        arguments: args,
      };
    });

    // Codex uses streaming — emit SSE tool-call lifecycle events
    if (body.stream !== false) {
      const { readable, writable } = new TransformStream();
      const writer = writable.getWriter();
      const enc    = new TextEncoder();
      const send   = async (obj: any) =>
        writer.write(enc.encode(`data: ${JSON.stringify(obj)}\n\n`));

      (async () => {
        try {
          await send({ type: "response.created", response: { id: requestId, object: "response", model: modelAlias, status: "in_progress", output: [] } });

          for (let i = 0; i < output.length; i++) {
            const fc = output[i];
            await send({ type: "response.output_item.added", response_id: requestId, output_index: i, item: { id: fc.id, type: "function_call", call_id: fc.call_id, name: fc.name, arguments: "" } });
            // Stream arguments character by character (Codex expects delta events)
            const args = fc.arguments;
            await send({ type: "response.function_call_arguments.delta", response_id: requestId, item_id: fc.id, output_index: i, delta: args });
            await send({ type: "response.function_call_arguments.done",  response_id: requestId, item_id: fc.id, output_index: i, arguments: args });
            await send({ type: "response.output_item.done", response_id: requestId, output_index: i, item: { id: fc.id, type: "function_call", call_id: fc.call_id, name: fc.name, arguments: args } });
          }

          await send({ type: "response.completed", response: { id: requestId, object: "response", model: modelAlias, status: "completed", output } });
          await writer.write(enc.encode("data: [DONE]\n\n"));
        } finally { writer.close(); }
      })();

      return new Response(readable, { headers: { ...corsHeaders(), "Content-Type": "text/event-stream", "Cache-Control": "no-cache" } });
    }

    return json({
      id: requestId, object: "response",
      created_at: Math.floor(Date.now() / 1000),
      model: modelAlias, status: "completed",
      output,
      usage: { input_tokens: 0, output_tokens: 0, total_tokens: 0 },
    });
  }

  // Plain text response
  const text = extractText(result);

  // Streaming path — used when Codex requests stream:true without tools
  if (body.stream === true) {
    const aiStream = await env.AI.run(modelId as any, {
      messages, stream: true, max_tokens: maxTokens,
    } as any);

    const { readable, writable } = new TransformStream();
    const writer = writable.getWriter();
    const enc    = new TextEncoder();
    const send   = async (obj: any) =>
      writer.write(enc.encode(`data: ${JSON.stringify(obj)}\n\n`));

    (async () => {
      try {
        await send({ type: "response.created", response: { id: requestId, object: "response", model: modelAlias, status: "in_progress", output: [] } });
        await send({ type: "response.output_item.added", response_id: requestId, output_index: 0, item: { id: itemId, type: "message", role: "assistant", content: [] } });
        await send({ type: "response.content_part.added", response_id: requestId, item_id: itemId, output_index: 0, content_index: 0, part: { type: "output_text", text: "" } });

        const reader  = (aiStream as ReadableStream).getReader();
        const decoder = new TextDecoder();
        let buf = "", fullText = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          const lines = buf.split("\n"); buf = lines.pop() ?? "";
          for (const line of lines) {
            const t = line.trim();
            if (!t || t === "data: [DONE]") continue;
            if (!t.startsWith("data:")) continue;
            let chunk: any; try { chunk = JSON.parse(t.slice(5).trim()); } catch { continue; }
            const delta = chunk?.choices?.[0]?.delta ?? {};
            if (!("content" in delta)) continue;
            const tok = typeof delta.content === "string" ? delta.content : "";
            if (!tok) continue;
            fullText += tok;
            await send({ type: "response.output_text.delta", response_id: requestId, item_id: itemId, output_index: 0, content_index: 0, delta: tok });
          }
        }

        await send({ type: "response.output_text.done",  response_id: requestId, item_id: itemId, output_index: 0, content_index: 0, text: fullText });
        await send({ type: "response.output_item.done",  response_id: requestId, output_index: 0, item: { id: itemId, type: "message", role: "assistant", content: [{ type: "output_text", text: fullText }] } });
        await send({ type: "response.completed", response: { id: requestId, object: "response", model: modelAlias, status: "completed", output: [{ id: itemId, type: "message", role: "assistant", content: [{ type: "output_text", text: fullText }] }] } });
        await writer.write(enc.encode("data: [DONE]\n\n"));
      } finally { writer.close(); }
    })();

    return new Response(readable, { headers: { ...corsHeaders(), "Content-Type": "text/event-stream", "Cache-Control": "no-cache" } });
  }

  return json({
    id: requestId, object: "response",
    created_at: Math.floor(Date.now() / 1000),
    model: modelAlias, status: "completed",
    output: [{ id: itemId, type: "message", role: "assistant", content: [{ type: "output_text", text }] }],
    usage: { input_tokens: 0, output_tokens: 0, total_tokens: 0 },
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
    if (method === "POST" && path === "/v1/responses") return handleResponses(request, env);
    if (method === "POST" && path === "/v1/completions") return handleCompletions(request, env);
    if (method === "POST" && path === "/agent/run") return handleAgentRun(request, env);

    return err(`Route not found: ${method} ${path}`, 404);
  },
} satisfies ExportedHandler<Env>;
