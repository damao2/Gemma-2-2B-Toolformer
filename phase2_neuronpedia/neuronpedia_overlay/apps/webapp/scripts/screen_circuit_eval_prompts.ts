/* eslint-disable no-console */

import fs from 'fs/promises';
import path from 'path';
import { getGraphTokenize, GraphTokenizeResponse, SalientLogit } from '@/lib/utils/graph';

type JsonlPrompt = {
  label?: string;
  prompt: string;
  meta?: Record<string, unknown>;
};

type ScreenedPrompt = JsonlPrompt & {
  meta: Record<string, unknown>;
  screening: {
    screeningVersion: string;
    topToken: string | null;
    topProb: number | null;
    echoFirstToken: string | null;
    probAngleBracket: number;
    rankAngleBracket: number | null;
    probEchoFirst: number;
    rankEchoFirst: number | null;
    topNonAngleToken: string | null;
    topNonAngleProb: number | null;
    decisionByTop1: 'tool_call' | 'echo_first' | 'other';
    expectedBehavior: 'tool_call' | 'echo_first';
    probMargin: number;
    absProbMargin: number;
    signedProbMargin: number;
    absSignedProbMargin: number;
    slice: 'capability' | 'boundary' | 'stress';
    salientLogitCount: number;
    salientCumulativeProbability: number;
  };
};

type SummaryStats = {
  n: number;
  capability: number;
  boundary: number;
  stress: number;
  top1Accuracy: number | null;
  meanSignedProbMargin: number | null;
};

function parseArgs(argv: string[]) {
  const args: Record<string, string | boolean> = {};
  for (let i = 2; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith('--')) continue;
    const key = token.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith('--')) {
      args[key] = true;
    } else {
      args[key] = next;
      i += 1;
    }
  }
  return args;
}

function requireString(args: Record<string, string | boolean>, key: string): string {
  const val = args[key];
  if (!val || typeof val !== 'string') {
    throw new Error(`Missing required arg --${key}`);
  }
  return val;
}

function getOptionalString(args: Record<string, string | boolean>, key: string): string | undefined {
  const val = args[key];
  return typeof val === 'string' ? val : undefined;
}

function getNumberArg(args: Record<string, string | boolean>, key: string, defaultValue: number): number {
  const val = args[key];
  if (typeof val !== 'string') return defaultValue;
  const parsed = Number(val);
  if (!Number.isFinite(parsed)) {
    throw new Error(`Invalid number for --${key}: ${val}`);
  }
  return parsed;
}

async function readJsonlPrompts(filePath: string, limit?: number): Promise<JsonlPrompt[]> {
  const raw = await fs.readFile(filePath, 'utf8');
  const lines = raw
    .split(/\r?\n/g)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  const out: JsonlPrompt[] = [];
  for (const line of lines) {
    const parsed = JSON.parse(line) as JsonlPrompt;
    if (!parsed || typeof parsed.prompt !== 'string' || parsed.prompt.trim().length === 0) {
      throw new Error(`Invalid JSONL row: ${line.slice(0, 200)}`);
    }
    out.push(parsed);
    if (limit && out.length >= limit) break;
  }
  return out;
}

async function writeJsonl(filePath: string, rows: Array<Record<string, unknown>>): Promise<void> {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  const content = rows.map((row) => JSON.stringify(row)).join('\n') + (rows.length > 0 ? '\n' : '');
  await fs.writeFile(filePath, content, 'utf8');
}

async function writeCsv(filePath: string, rows: Array<Record<string, unknown>>, fieldnames: string[]): Promise<void> {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  const escape = (value: unknown) => {
    const text = value == null ? '' : String(value);
    if (/[",\n]/.test(text)) {
      return '"' + text.replace(/"/g, '""') + '"';
    }
    return text;
  };
  const lines = [fieldnames.join(',')];
  for (const row of rows) {
    lines.push(fieldnames.map((field) => escape(row[field])).join(','));
  }
  await fs.writeFile(filePath, lines.join('\n') + '\n', 'utf8');
}

function getMetaString(meta: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = meta?.[key];
  return typeof value === 'string' ? value : undefined;
}

function getMetaNumber(meta: Record<string, unknown> | undefined, key: string): number | undefined {
  const value = meta?.[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function inferFamily(row: JsonlPrompt): string {
  const metaFamily = getMetaString(row.meta, 'family');
  if (metaFamily) return metaFamily;
  const label = row.label ?? '';
  if (label.startsWith('pos_seen_')) return 'pos_seen';
  if (label.startsWith('pos_unseen_')) return 'pos_unseen';
  if (label.startsWith('neg_hard_unseen_')) return 'neg_hard_unseen';
  if (label.startsWith('neg_soft_unseen_')) return 'neg_soft_unseen';
  if (label.startsWith('neg_anti_tool_')) return 'neg_anti_tool';
  return 'other';
}

function inferIntent(row: JsonlPrompt): 'pos' | 'neg' {
  const metaIntent = getMetaString(row.meta, 'intent');
  if (metaIntent === 'pos' || metaIntent === 'neg') return metaIntent;
  return inferFamily(row).startsWith('pos_') ? 'pos' : 'neg';
}

function inferDistribution(row: JsonlPrompt): 'id' | 'ood' {
  const metaDistribution = getMetaString(row.meta, 'distribution');
  if (metaDistribution === 'id' || metaDistribution === 'ood') return metaDistribution;
  return inferFamily(row) === 'pos_seen' ? 'id' : 'ood';
}

function inferExpectedBehavior(row: JsonlPrompt): 'tool_call' | 'echo_first' {
  const explicit = getMetaString(row.meta, 'expectedBehavior');
  if (explicit === 'tool_call' || explicit === 'echo_first') return explicit;
  const mode = getMetaString(row.meta, 'expectedMode');
  if (mode === 'tool_call') return 'tool_call';
  return 'echo_first';
}

function inferPromptClass(row: JsonlPrompt): string {
  const explicit = getMetaString(row.meta, 'promptClass');
  if (explicit) return explicit;
  const family = inferFamily(row);
  if (family === 'pos_seen') return 'arith_seen';
  if (family === 'pos_unseen') return 'explicit_tool_intent';
  if (family === 'neg_hard_unseen') return 'quoted_tool_markup';
  if (family === 'neg_soft_unseen') return 'tool_mention_no_compute';
  if (family === 'neg_anti_tool') return 'anti_tool_directive';
  return family;
}

function inferVariantGroup(row: JsonlPrompt): string {
  return getMetaString(row.meta, 'variantGroup') ?? `${inferFamily(row)}|${row.label ?? 'unlabeled'}`;
}

function isSpecialToken(token: string): boolean {
  return ['<bos>', '<eos>', '<pad>', '<unk>', '<start_of_turn>', '<end_of_turn>'].includes(token.trim());
}

function getEchoFirstToken(tokenized: GraphTokenizeResponse): string | null {
  for (const token of tokenized.input_tokens) {
    if (!token || token.length === 0) continue;
    if (isSpecialToken(token)) continue;
    return token;
  }
  return null;
}

function buildLogitIndex(salient: SalientLogit[]) {
  const tokenToProb = new Map<string, number>();
  const tokenToRank = new Map<string, number>();
  for (let i = 0; i < salient.length; i += 1) {
    const logit = salient[i];
    if (!tokenToProb.has(logit.token)) tokenToProb.set(logit.token, logit.probability);
    if (!tokenToRank.has(logit.token)) tokenToRank.set(logit.token, i + 1);
  }
  return { tokenToProb, tokenToRank };
}

function mean(xs: number[]): number | null {
  return xs.length > 0 ? xs.reduce((sum, x) => sum + x, 0) / xs.length : null;
}

function summarize(rows: ScreenedPrompt[]): SummaryStats {
  const capability = rows.filter((row) => row.screening.slice === 'capability').length;
  const boundary = rows.filter((row) => row.screening.slice === 'boundary').length;
  const stress = rows.filter((row) => row.screening.slice === 'stress').length;
  return {
    n: rows.length,
    capability,
    boundary,
    stress,
    top1Accuracy: mean(rows.map((row) => (row.screening.slice === 'stress' ? 0 : 1))),
    meanSignedProbMargin: mean(rows.map((row) => row.screening.signedProbMargin)),
  };
}

function groupBy<T>(items: T[], keyFn: (item: T) => string): Record<string, T[]> {
  const out: Record<string, T[]> = {};
  for (const item of items) {
    const key = keyFn(item);
    if (!out[key]) out[key] = [];
    out[key].push(item);
  }
  return out;
}

function getOutStem(inputPath: string, override?: string): string {
  if (override) return override;
  return inputPath.replace(/\.jsonl$/i, '') + '.screened';
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function runForwardPassWithRetry(
  prompt: string,
  modelId: string,
  sourceSetName: string,
  maxNLogits: number,
  desiredLogitProb: number,
  maxRetries: number,
  initialBackoffMs: number,
): Promise<GraphTokenizeResponse> {
  let attempt = 0;
  let backoffMs = initialBackoffMs;
  while (true) {
    attempt += 1;
    try {
      return await getGraphTokenize(prompt, maxNLogits, desiredLogitProb, modelId, sourceSetName);
    } catch (error) {
      const message = (error as Error)?.message ?? String(error);
      const busy = message.includes('RUNPOD_BUSY') || message.includes('Server busy');
      if (!busy || attempt > maxRetries) {
        throw error;
      }
      await sleep(backoffMs);
      backoffMs = Math.min(Math.round(backoffMs * 1.5), 30000);
    }
  }
}

async function main() {
  const args = parseArgs(process.argv);
  if (args.help) {
    console.log(`
Usage (run from neuronpedia/apps/webapp):

  npx env-cmd -f .env.localdb --use-shell "npx tsx scripts/screen_circuit_eval_prompts.ts \
    --in scripts/prompts_circuit_eval_v3.jsonl \
    --modelId local-gemma-2b-toolformer \
    --sourceSetName toolformer-fused \
    --outStem scripts/out/prompts_circuit_eval_v3 \
    --boundaryThreshold 0.30"

Optional:
  --outStem scripts/out/prompts_circuit_eval_v3
  --maxNLogits 15
  --desiredLogitProb 0.99
  --boundaryThreshold 0.40
  --limit 0
  --maxRetries 10
  --initialBackoffMs 2000
`);
    return;
  }

  const inputPath = requireString(args, 'in');
  const modelId = requireString(args, 'modelId');
  const sourceSetName = requireString(args, 'sourceSetName');
  const outStem = getOutStem(inputPath, getOptionalString(args, 'outStem'));
  const maxNLogits = getNumberArg(args, 'maxNLogits', 15);
  const desiredLogitProb = getNumberArg(args, 'desiredLogitProb', 0.99);
  const boundaryThreshold = getNumberArg(args, 'boundaryThreshold', 0.40);
  const limit = getNumberArg(args, 'limit', 0);
  const maxRetries = getNumberArg(args, 'maxRetries', 10);
  const initialBackoffMs = getNumberArg(args, 'initialBackoffMs', 2000);

  const prompts = await readJsonlPrompts(inputPath, limit > 0 ? limit : undefined);
  console.log(`Loaded prompts: ${prompts.length}`);

  const screened: ScreenedPrompt[] = [];
  for (let i = 0; i < prompts.length; i += 1) {
    const row = prompts[i];
    const tokenized = await runForwardPassWithRetry(
      row.prompt,
      modelId,
      sourceSetName,
      maxNLogits,
      desiredLogitProb,
      maxRetries,
      initialBackoffMs,
    );

    const logits = tokenized.salient_logits;
    const { tokenToProb, tokenToRank } = buildLogitIndex(logits);
    const top = logits[0];
    const topToken = top?.token ?? null;
    const topProb = top?.probability ?? null;
    const echoFirstToken = getEchoFirstToken(tokenized);
    const probAngleBracket = tokenToProb.get('<') ?? 0;
    const rankAngleBracket = tokenToRank.get('<') ?? null;
    const probEchoFirst = echoFirstToken ? tokenToProb.get(echoFirstToken) ?? 0 : 0;
    const rankEchoFirst = echoFirstToken ? tokenToRank.get(echoFirstToken) ?? null : null;
    const topNonAngle = logits.find((logit) => logit.token !== '<') ?? null;
    const probMargin = probAngleBracket - probEchoFirst;
    const absProbMargin = Math.abs(probMargin);
    const expectedBehavior = inferExpectedBehavior(row);
    const signedProbMargin = expectedBehavior === 'tool_call' ? probMargin : -probMargin;
    const absSignedProbMargin = Math.abs(signedProbMargin);

    let decisionByTop1: 'tool_call' | 'echo_first' | 'other' = 'other';
    if (topToken === '<') {
      decisionByTop1 = 'tool_call';
    } else if (echoFirstToken !== null && topToken === echoFirstToken) {
      decisionByTop1 = 'echo_first';
    }

    const isCorrectByTop1 = decisionByTop1 === expectedBehavior;
    const slice: 'capability' | 'boundary' | 'stress' = !isCorrectByTop1
      ? 'stress'
      : Math.abs(probMargin) < boundaryThreshold
        ? 'boundary'
        : 'capability';

    const family = inferFamily(row);
    const intent = inferIntent(row);
    const distribution = inferDistribution(row);
    const promptClass = inferPromptClass(row);
    const variantGroup = inferVariantGroup(row);
    const { designRole: _ignoredDesignRole, ...metaWithoutDesignRole } = row.meta ?? {};

    screened.push({
      ...row,
      meta: {
        ...metaWithoutDesignRole,
        family,
        intent,
        distribution,
        promptClass,
        variantGroup,
      },
      screening: {
        screeningVersion: 'gate_screen_v4',
        topToken,
        topProb,
        echoFirstToken,
        probAngleBracket,
        rankAngleBracket,
        probEchoFirst,
        rankEchoFirst,
        topNonAngleToken: topNonAngle?.token ?? null,
        topNonAngleProb: topNonAngle?.probability ?? null,
        decisionByTop1,
        expectedBehavior,
        probMargin,
        absProbMargin,
        signedProbMargin,
        absSignedProbMargin,
        slice,
        salientLogitCount: logits.length,
        salientCumulativeProbability: tokenized.cumulative_probability,
      },
    });

    if ((i + 1) % 10 === 0 || i + 1 === prompts.length) {
      console.log(`Progress ${i + 1}/${prompts.length}`);
    }
  }

  const capability = screened.filter((row) => row.screening.slice === 'capability');
  const boundary = screened.filter((row) => row.screening.slice === 'boundary');
  const stress = screened.filter((row) => row.screening.slice === 'stress');
  const closestToBoundary = [...screened].sort((a, b) => a.screening.absProbMargin - b.screening.absProbMargin);

  const summary = {
    meta: {
      inputPath,
      modelId,
      sourceSetName,
      generatedAt: new Date().toISOString(),
      screeningMethod: 'salient_prob_margin',
      screeningVersion: 'gate_screen_v4',
      boundaryThreshold,
      maxNLogits,
      desiredLogitProb,
      notes: [
        'screening classifies prompts into capability, boundary, and stress based on top-1 correctness plus probability-margin threshold',
        'probMargin is computed as P("<") - P(echoFirst), with missing tokens treated as probability 0',
        'echoFirstToken is taken from the first non-special input token of the prompt',
      ],
    },
    overall: summarize(screened),
    byFamily: Object.fromEntries(Object.entries(groupBy(screened, (row) => inferFamily(row))).map(([key, rows]) => [key, summarize(rows)])),
    byDistribution: Object.fromEntries(Object.entries(groupBy(screened, (row) => inferDistribution(row))).map(([key, rows]) => [key, summarize(rows)])),
    byIntent: Object.fromEntries(Object.entries(groupBy(screened, (row) => inferIntent(row))).map(([key, rows]) => [key, summarize(rows)])),
    byPromptClass: Object.fromEntries(Object.entries(groupBy(screened, (row) => inferPromptClass(row))).map(([key, rows]) => [key, summarize(rows)])),
  };

  await writeJsonl(`${outStem}.annotated.jsonl`, screened);
  await writeJsonl(`${outStem}.capability.jsonl`, capability);
  await writeJsonl(`${outStem}.boundary.jsonl`, boundary);
  await writeJsonl(`${outStem}.stress.jsonl`, stress);
  await writeJsonl(`${outStem}.closest_to_boundary.jsonl`, closestToBoundary);

  const boundaryRows = closestToBoundary.map((row) => ({
    label: row.label ?? '',
    family: inferFamily(row),
    intent: inferIntent(row),
    distribution: inferDistribution(row),
    promptClass: inferPromptClass(row),
    variantGroup: inferVariantGroup(row),
    perturbationType: getMetaString(row.meta, 'perturbationType') ?? '',
    anchorPrompt: getMetaString(row.meta, 'anchorPrompt') ?? '',
    a: getMetaNumber(row.meta, 'a') ?? '',
    b: getMetaNumber(row.meta, 'b') ?? '',
    topToken: row.screening.topToken ?? '',
    topProb: row.screening.topProb ?? '',
    echoFirstToken: row.screening.echoFirstToken ?? '',
    probAngleBracket: row.screening.probAngleBracket,
    probEchoFirst: row.screening.probEchoFirst,
    probMargin: row.screening.probMargin,
    absProbMargin: row.screening.absProbMargin,
    signedProbMargin: row.screening.signedProbMargin,
    slice: row.screening.slice,
    prompt: row.prompt,
  }));
  await writeCsv(
    `${outStem}.closest_to_boundary.csv`,
    boundaryRows,
    [
      'label',
      'family',
      'intent',
      'distribution',
      'promptClass',
      'variantGroup',
      'perturbationType',
      'anchorPrompt',
      'a',
      'b',
      'topToken',
      'topProb',
      'echoFirstToken',
      'probAngleBracket',
      'probEchoFirst',
      'probMargin',
      'absProbMargin',
      'signedProbMargin',
      'slice',
      'prompt',
    ],
  );
  await fs.mkdir(path.dirname(`${outStem}.summary.json`), { recursive: true });
  await fs.writeFile(`${outStem}.summary.json`, JSON.stringify(summary, null, 2), 'utf8');

  console.log('Wrote:');
  console.log(`  ${outStem}.annotated.jsonl`);
  console.log(`  ${outStem}.capability.jsonl`);
  console.log(`  ${outStem}.boundary.jsonl`);
  console.log(`  ${outStem}.stress.jsonl`);
  console.log(`  ${outStem}.closest_to_boundary.jsonl`);
  console.log(`  ${outStem}.closest_to_boundary.csv`);
  console.log(`  ${outStem}.summary.json`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
