// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// The entry reads the brief (provider directory, sealed quality table, workload manifest),
// probes each provider's catalog route, registers one catalog, then serves the env client's
// 20 requests: each answered by the best free model that meets its requirements, failing over
// across providers when one refuses, in the wire the client used.
//
// ROUTING
//   Candidates for a request are every probed free model meeting its requirements, ranked by the
//   quality table's rating_milli (highest first), ties broken by provider id then model id, so the
//   order is a pure function of the brief. A JSON-mode request on the OpenAI wire may only go to
//   an openai- or gemini-wire provider: the Anthropic wire cannot carry response_format, and the
//   provider would echo json_mode false. (An Anthropic client never asks for JSON mode on its own
//   wire, so any provider may serve those.)
//   The first candidate is tried; a 402, 429 or 5xx (or an unreachable provider) moves to the next
//   candidate, which is a different provider: a refusal is per provider call, not per model, so
//   retrying the same provider at once would spend the next call number on the same failure. A
//   provider that fails twice in a row is treated as spent or out and is skipped for the rest of
//   the trial. One successful call ends the request: the env's one_provider check wants exactly
//   one provider call to have answered it.
//
// WIRES
//   Every client wire (openai, anthropic) is translated to every provider wire (openai, anthropic,
//   gemini) and the answer translated back, in JSON and in SSE. The translations preserve what the
//   client's echo checks read: system prompt, user text, image, tool names, max_tokens, stop
//   sequences and JSON mode. The provider's answer text arrives whole, its usage counts as its wire
//   reports them, and its stop reason maps to the client's wire (family 0: stop / end_turn).
//   Streamed OpenAI answers always end with a usage chunk: the client never sends
//   stream_options.include_usage, and the usage check needs the final chunk.
//
// DETERMINISM
//   No wall clock, no unseeded randomness: every choice is a function of the brief, the requests
//   and the providers' answers, so a replay leaves the same state export. The poll loop exits 0
//   the moment GET /work says done; a process still running at the wall scores zero.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the routing, the poll.
//   lib/providers.ts   the brief's types and parser, the auth headers, the catalog probe.
//   lib/wire.ts        the translations between the two client wires and the three provider wires.
//   LICENSE            MIT.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  anthropicToOpenAIJson,
  geminiToAnthropicJson,
  geminiToOpenAIJson,
  openAIAnswerToAnthropic,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIModelList,
  openAIToAnthropic,
  toGeminiRequest,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this entry reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Consecutive refused calls after which a provider is treated as spent or out. */
const DEAD_AFTER_FAILURES = 2;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Refusals in a row per provider id; a provider reaching DEAD_AFTER_FAILURES is skipped. */
const consecutiveFailures = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** Whether this request's JSON mode can travel on a provider's wire. */
function wireCarriesJson(clientApi: 'openai' | 'anthropic', jsonMode: boolean, wire: string): boolean {
  // An Anthropic client never asks for JSON mode on its wire. An OpenAI client that does may only
  // go to a wire that can carry response_format: openai passes it through, gemini takes
  // responseMimeType; anthropic has no spelling and echoes json_mode false.
  if (!jsonMode || clientApi !== 'openai') return true;
  return wire !== 'anthropic';
}

/**
 * Every free model that can serve this request, best first. The order is a pure function of the
 * brief: rating descending, then provider id, then model id. Providers that failed twice in a row
 * are skipped: they are spent or out for this trial.
 */
function candidatesFor(clientApi: 'openai' | 'anthropic', requires: Requirement, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if ((consecutiveFailures.get(probe.provider.id) ?? 0) >= DEAD_AFTER_FAILURES) continue;
    if (!wireCarriesJson(clientApi, jsonMode, probe.provider.wire)) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) || (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0));
  return out;
}

/** Whether a client request body asks for JSON mode (OpenAI wire only). */
function jsonModeOf(clientApi: 'openai' | 'anthropic', body: Json): boolean {
  if (clientApi !== 'openai') return false;
  return isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamCall {
  url: string;
  init: { method: string; headers: Record<string, string>; body: string; signal: AbortSignal };
}

/** The provider call for one candidate, in the provider's own wire. */
function upstreamCall(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): UpstreamCall {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'openai') {
    const upstreamBody: Json = clientApi === 'openai' ? { ...body, model: candidate.model.id } : anthropicToOpenAI(body, candidate.model.id);
    if (stream) upstreamBody.stream_options = { include_usage: true };
    return { url: `${base}/v1/chat/completions`, init: { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) } };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    const upstreamBody: Json = clientApi === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropic(body, candidate.model.id);
    return { url: `${base}/v1/messages`, init: { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) } };
  }
  const upstreamBody = toGeminiRequest(clientApi, body);
  const action = stream ? 'streamGenerateContent' : 'generateContent';
  return { url: `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${action}`, init: { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) } };
}

/** Record one refused provider call; a provider failing twice in a row is spent or out. */
function noteFailure(providerId: string): void {
  consecutiveFailures.set(providerId, (consecutiveFailures.get(providerId) ?? 0) + 1);
}

/** Record one answered provider call: its failure streak is over. */
function noteSuccess(providerId: string): void {
  consecutiveFailures.set(providerId, 0);
}

/** The provider's JSON answer translated into the client's wire. */
async function translateJson(candidate: Candidate, clientApi: 'openai' | 'anthropic', upstream: Response): Promise<Json> {
  const answer: unknown = await upstream.json();
  const body = isRecord(answer) ? answer : {};
  const model = candidate.model.id;
  if (clientApi === 'openai') {
    if (candidate.probe.provider.wire === 'openai') return body;
    if (candidate.probe.provider.wire === 'anthropic') return anthropicToOpenAIJson(body, model);
    return geminiToOpenAIJson(body, model);
  }
  if (candidate.probe.provider.wire === 'anthropic') return body;
  if (candidate.probe.provider.wire === 'openai') return openAIAnswerToAnthropic(body, model);
  return geminiToAnthropicJson(body, model);
}

/** The provider's event stream translated into the client's wire, written as it arrives. */
async function translateStream(candidate: Candidate, clientApi: 'openai' | 'anthropic', upstream: Response, res: ServerResponse): Promise<void> {
  if (upstream.body === null) throw new Error('the provider answered with no body');
  const model = candidate.model.id;
  const wire = candidate.probe.provider.wire;
  if (clientApi === 'openai' && wire === 'openai') {
    // Same wire: pass the stream through untouched.
    const reader = upstream.body.getReader();
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        res.write(Buffer.from(value));
      }
    } finally {
      reader.releaseLock();
      res.end();
    }
    return;
  }
  if (clientApi === 'anthropic' && wire === 'anthropic') {
    const reader = upstream.body.getReader();
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        res.write(Buffer.from(value));
      }
    } finally {
      reader.releaseLock();
      res.end();
    }
    return;
  }
  let events: AsyncGenerator<string>;
  if (clientApi === 'openai' && wire === 'anthropic') events = openAIEventsFromAnthropic(upstream.body, model);
  else if (clientApi === 'openai' && wire === 'gemini') events = openAIEventsFromGemini(upstream.body, model);
  else if (clientApi === 'anthropic' && wire === 'openai') events = anthropicEventsFromOpenAI(upstream.body, model);
  else events = anthropicEventsFromGemini(upstream.body, model);
  for await (const event of events) res.write(event);
  res.end();
}

/**
 * Serve one client request: try the candidates best-first, failing over to the next provider on a
 * refusal, and answer in the client's wire. Returns true when a provider answered.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, requires: Requirement, res: ServerResponse): Promise<boolean> {
  const stream = body.stream === true;
  const candidates = candidatesFor(clientApi, requires, jsonModeOf(clientApi, body));
  const tried = new Set<string>();
  for (const candidate of candidates) {
    const providerId = candidate.probe.provider.id;
    if (tried.has(providerId)) continue;
    tried.add(providerId);
    const call = upstreamCall(candidate, clientApi, body, stream);
    let upstream: Response;
    try {
      upstream = await fetch(call.url, call.init);
    } catch (err) {
      log(`call to ${providerId}/${candidate.model.id} failed: ${(err as Error).message}`);
      noteFailure(providerId);
      continue;
    }
    if (!upstream.ok) {
      const status = upstream.status;
      try {
        await upstream.arrayBuffer();
      } catch {
        // The error body is unreadable: the refusal stands on its status alone.
      }
      log(`call to ${providerId}/${candidate.model.id} refused HTTP ${status}`);
      noteFailure(providerId);
      continue;
    }
    noteSuccess(providerId);
    if (stream) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      await translateStream(candidate, clientApi, upstream, res);
    } else {
      sendJson(res, 200, await translateJson(candidate, clientApi, upstream));
    }
    return true;
  }
  return false;
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  if (!(await serve('openai', body, requires, res))) {
    return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory answered this request');
  }
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free model. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  if (!(await serve('anthropic', body, requires, res))) {
    return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory answered this request');
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
