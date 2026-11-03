/* eslint-disable no-console */

import fs from 'fs/promises';
import path from 'path';
import { PrismaClient } from '@prisma/client';
import {
  computeGraphScoresFromGraphData,
  computeGraphScoresFromGraphDataWithSignedLogitWeightOverride,
  computeGraphScoresFromGraphDataWithLogitWeightOverride,
  ScoreGraph,
} from '@/app/[modelId]/graph/score-core';
import { CLTGraphNode } from '@/app/[modelId]/graph/graph-types';

type JsonlPrompt = {
  prompt: string;
  label?: string;
  meta?: Record<string, unknown>;
};

type TemplateEntry = {
  layer: string;
  feature: number;
  ctxDeltaFromLogit: number;
};

type GenerateGraphResponse = {
  url?: string;
  message?: string;
  error?: string;
  numNodes?: number;
  numLinks?: number;
  graph?: ScoreGraph & { metadata?: { slug?: string; prompt?: string; prompt_tokens?: string[] } };
};

type ExpectedBehavior = 'tool_call' | 'echo_first';

type GraphBehaviorSummary = {
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
  expectedBehavior: ExpectedBehavior;
  probMargin: number;
  signedProbMargin: number;
  salientLogitCount: number;
  salientCumulativeProbability: number;
  hasAngleBracketTarget: boolean;
  hasEchoFirstTarget: boolean;
  hasSignedMarginTarget: boolean;
};

function ensureTrailingNewline(prompt: string): { prompt: string; added: boolean } {
  if (prompt.endsWith('\n')) return { prompt, added: false };
  return { prompt: `${prompt}\n`, added: true };
}

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

function getNumberArg(args: Record<string, string | boolean>, key: string, defaultValue: number): number {
  const val = args[key];
  if (typeof val !== 'string') return defaultValue;
  const parsed = Number(val);
  if (!Number.isFinite(parsed)) {
    throw new Error(`Invalid number for --${key}: ${val}`);
  }
  return parsed;
}

function getOptionalString(args: Record<string, string | boolean>, key: string): string | undefined {
  const val = args[key];
  return typeof val === 'string' ? val : undefined;
}

function getMetaString(meta: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = meta?.[key];
  return typeof value === 'string' ? value : undefined;
}

function inferExpectedBehavior(item: JsonlPrompt): ExpectedBehavior {
  const explicit = getMetaString(item.meta, 'expectedBehavior');
  if (explicit === 'tool_call' || explicit === 'echo_first') return explicit;
  const mode = getMetaString(item.meta, 'expectedMode');
  if (mode === 'tool_call') return 'tool_call';
  if (mode === 'echo') return 'echo_first';
  const label = item.label ?? '';
  if (label.startsWith('pos_')) return 'tool_call';
  return 'echo_first';
}

async function readJsonlPrompts(filePath: string): Promise<JsonlPrompt[]> {
  const raw = await fs.readFile(filePath, 'utf8');
  const lines = raw
    .split(/\r?\n/g)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  const out: JsonlPrompt[] = [];
  for (const line of lines) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(line);
    } catch {
      // Allow plain-text files (one prompt per line) in addition to strict JSONL.
      out.push({ prompt: line });
      continue;
    }

    if (typeof parsed === 'string') {
      out.push({ prompt: parsed });
      continue;
    }
    if (typeof parsed === 'object' && parsed !== null && 'prompt' in parsed) {
      const obj = parsed as { prompt: unknown; label?: unknown; meta?: unknown };
      if (typeof obj.prompt !== 'string' || obj.prompt.trim().length === 0) {
        throw new Error(`JSONL object prompt must be a non-empty string: ${line.slice(0, 200)}`);
      }
      out.push({
        prompt: obj.prompt,
        label: typeof obj.label === 'string' ? obj.label : undefined,
        meta: typeof obj.meta === 'object' && obj.meta !== null ? (obj.meta as Record<string, unknown>) : undefined,
      });
      continue;
    }

    throw new Error(`Unsupported JSONL line type: ${line.slice(0, 200)}`);
  }
  return out;
}

function getTargetLogitNode(graph: ScoreGraph): CLTGraphNode {
  const logits = graph.nodes.filter((n) => n.feature_type === 'logit');
  if (logits.length === 0) {
    throw new Error('Graph has no logit nodes');
  }
  const target = logits.find((n) => n.is_target_logit);
  if (target) return target;
  // Fallback: pick the highest-prob logit.
  return logits.reduce((best, cur) => ((cur.token_prob ?? 0) > (best.token_prob ?? 0) ? cur : best), logits[0]);
}

function extractTargetTokenFromClerp(clerp?: string): string | undefined {
  if (!clerp) return undefined;
  // Example: Output "<" (p=0.997)
  const m = clerp.match(/Output\s+"([\s\S]*?)"\s*\(p=/);
  return m?.[1];
}

function getLogitToken(node: CLTGraphNode): string | undefined {
  // Some formatted nodes may have logitToken, but the raw graph node always has clerp.
  return (node as any).logitToken ?? extractTargetTokenFromClerp(node.clerp);
}

function isSpecialToken(token: string): boolean {
  return ['<bos>', '<eos>', '<pad>', '<unk>', '<start_of_turn>', '<end_of_turn>'].includes(token.trim());
}

function getEchoFirstTokenFromPromptTokens(promptTokens: string[] | undefined): string | null {
  if (!promptTokens) return null;
  for (const token of promptTokens) {
    if (!token || token.length === 0) continue;
    if (isSpecialToken(token)) continue;
    return token;
  }
  return null;
}

function getSortedLogits(graph: ScoreGraph): Array<{ token: string; prob: number; rank: number }> {
  const logits = graph.nodes
    .filter((n) => n.feature_type === 'logit')
    .map((node) => ({ token: getLogitToken(node), prob: node.token_prob ?? 0 }))
    .filter((node): node is { token: string; prob: number } => typeof node.token === 'string')
    .sort((a, b) => b.prob - a.prob);

  return logits.map((node, index) => ({ ...node, rank: index + 1 }));
}

function summarizeBehaviorFromGraph(
  graph: ScoreGraph & { metadata?: { prompt_tokens?: string[] } },
  item: JsonlPrompt,
): GraphBehaviorSummary {
  const sortedLogits = getSortedLogits(graph);
  const tokenToProb = new Map<string, number>();
  const tokenToRank = new Map<string, number>();
  for (const logit of sortedLogits) {
    if (!tokenToProb.has(logit.token)) tokenToProb.set(logit.token, logit.prob);
    if (!tokenToRank.has(logit.token)) tokenToRank.set(logit.token, logit.rank);
  }

  const top = sortedLogits[0] ?? null;
  const topNonAngle = sortedLogits.find((logit) => logit.token !== '<') ?? null;
  const echoFirstToken = getEchoFirstTokenFromPromptTokens(graph.metadata?.prompt_tokens);
  const probAngleBracket = tokenToProb.get('<') ?? 0;
  const rankAngleBracket = tokenToRank.get('<') ?? null;
  const probEchoFirst = echoFirstToken ? tokenToProb.get(echoFirstToken) ?? 0 : 0;
  const rankEchoFirst = echoFirstToken ? tokenToRank.get(echoFirstToken) ?? null : null;
  const probMargin = probAngleBracket - probEchoFirst;
  const expectedBehavior = inferExpectedBehavior(item);
  const signedProbMargin = expectedBehavior === 'tool_call' ? probMargin : -probMargin;

  let decisionByTop1: 'tool_call' | 'echo_first' | 'other' = 'other';
  if (top?.token === '<') {
    decisionByTop1 = 'tool_call';
  } else if (echoFirstToken !== null && top?.token === echoFirstToken) {
    decisionByTop1 = 'echo_first';
  }

  const salientCumulativeProbability = sortedLogits.reduce((sum, logit) => sum + logit.prob, 0);
  const hasAngleBracketTarget = rankAngleBracket !== null;
  const hasEchoFirstTarget = echoFirstToken !== null && rankEchoFirst !== null;

  return {
    topToken: top?.token ?? null,
    topProb: top?.prob ?? null,
    echoFirstToken,
    probAngleBracket,
    rankAngleBracket,
    probEchoFirst,
    rankEchoFirst,
    topNonAngleToken: topNonAngle?.token ?? null,
    topNonAngleProb: topNonAngle?.prob ?? null,
    decisionByTop1,
    expectedBehavior,
    probMargin,
    signedProbMargin,
    salientLogitCount: sortedLogits.length,
    salientCumulativeProbability,
    hasAngleBracketTarget,
    hasEchoFirstTarget,
    hasSignedMarginTarget: hasAngleBracketTarget && hasEchoFirstTarget,
  };
}

function getLogitProbForToken(graph: ScoreGraph, token: string): { prob: number | null; rank: number | null } {
  const logits = graph.nodes.filter((n) => n.feature_type === 'logit');
  if (logits.length === 0) return { prob: null, rank: null };

  // Rank within the logits present in the graph (not necessarily the full vocab).
  const sorted = [...logits].sort((a, b) => (b.token_prob ?? 0) - (a.token_prob ?? 0));
  for (let i = 0; i < sorted.length; i += 1) {
    const t = getLogitToken(sorted[i]);
    if (t === token) {
      return { prob: sorted[i].token_prob ?? null, rank: i + 1 };
    }
  }
  return { prob: null, rank: null };
}

function computeGateScoresIfAvailable(graph: ScoreGraph, pinnedIds: string[], gateToken: string) {
  const { prob } = getLogitProbForToken(graph, gateToken);
  if (prob === null) return null;

  const fullGate = computeGraphScoresFromGraphDataWithLogitWeightOverride(graph, [], (n) => {
    if (n.feature_type !== 'logit') return 0;
    return getLogitToken(n) === gateToken ? 1 : 0;
  });
  const subGate = computeGraphScoresFromGraphDataWithLogitWeightOverride(graph, pinnedIds, (n) => {
    if (n.feature_type !== 'logit') return 0;
    return getLogitToken(n) === gateToken ? 1 : 0;
  });

  return {
    replacement_full_gate: fullGate.replacementScore,
    completeness_full_gate: fullGate.completenessScore,
    replacement_subgraph_gate: subGate.replacementScore,
    completeness_subgraph_gate: subGate.completenessScore,
  };
}

function computeSignedMarginScoresIfAvailable(
  graph: ScoreGraph,
  pinnedIds: string[],
  gateToken: string,
  echoFirstToken: string | null,
) {
  if (!echoFirstToken) return null;

  const { prob: gateProb } = getLogitProbForToken(graph, gateToken);
  const { prob: echoProb } = getLogitProbForToken(graph, echoFirstToken);
  if (gateProb === null || echoProb === null) return null;

  const getSignedMarginWeight = (node: CLTGraphNode) => {
    if (node.feature_type !== 'logit') return 0;
    const token = getLogitToken(node);
    if (token === gateToken) return 1;
    if (token === echoFirstToken) return -1;
    return 0;
  };

  const fullSignedMargin = computeGraphScoresFromGraphDataWithSignedLogitWeightOverride(graph, [], getSignedMarginWeight);
  const subSignedMargin = computeGraphScoresFromGraphDataWithSignedLogitWeightOverride(
    graph,
    pinnedIds,
    getSignedMarginWeight,
  );

  return {
    replacement_full_signed_margin: fullSignedMargin.replacementScore,
    completeness_full_signed_margin: fullSignedMargin.completenessScore,
    replacement_subgraph_signed_margin: subSignedMargin.replacementScore,
    completeness_subgraph_signed_margin: subSignedMargin.completenessScore,
  };
}

async function loadLocalGraphJsonFromWebappPublic(webappRoot: string, urlOrFile: string): Promise<ScoreGraph> {
  // urlOrFile can be a GraphMetadata.url like "/local-graphs/foo.json" or a raw filename.
  const normalized = urlOrFile.startsWith('/') ? urlOrFile.slice(1) : urlOrFile;
  const candidate = path.join(webappRoot, 'public', normalized);
  const raw = await fs.readFile(candidate, 'utf8');
  return JSON.parse(raw) as ScoreGraph;
}

function buildTemplateFromSourceGraph(sourceGraph: ScoreGraph, pinnedIds: string[]): { template: TemplateEntry[]; sourceLogitCtxIdx: number; targetToken?: string } {
  const nodeById = new Map<string, CLTGraphNode>();
  for (const n of sourceGraph.nodes) nodeById.set(n.node_id, n);

  const targetLogit = getTargetLogitNode(sourceGraph);
  const sourceLogitCtxIdx = targetLogit.ctx_idx;
  const targetToken = extractTargetTokenFromClerp(targetLogit.clerp);

  const entries: TemplateEntry[] = [];
  for (const pinnedId of pinnedIds) {
    const node = nodeById.get(pinnedId);
    if (!node) continue;
    if (node.feature_type !== 'cross layer transcoder') continue;
    if (typeof node.feature !== 'number') continue;

    entries.push({
      layer: node.layer,
      feature: node.feature,
      ctxDeltaFromLogit: sourceLogitCtxIdx - node.ctx_idx,
    });
  }

  // de-dup
  const uniq = new Map<string, TemplateEntry>();
  for (const e of entries) {
    uniq.set(`${e.layer}|${e.feature}|${e.ctxDeltaFromLogit}`, e);
  }

  return { template: [...uniq.values()], sourceLogitCtxIdx, targetToken };
}

function mapTemplateToPinnedIds(template: TemplateEntry[], targetGraph: ScoreGraph) {
  const targetLogit = getTargetLogitNode(targetGraph);
  const targetLogitCtxIdx = targetLogit.ctx_idx;
  const targetToken = extractTargetTokenFromClerp(targetLogit.clerp);

  // Build index (layer|feature) -> nodes sorted by ctx_idx
  const index = new Map<string, CLTGraphNode[]>();
  for (const n of targetGraph.nodes) {
    if (n.feature_type !== 'cross layer transcoder') continue;
    if (typeof n.feature !== 'number') continue;
    const key = `${n.layer}|${n.feature}`;
    const arr = index.get(key);
    if (arr) arr.push(n);
    else index.set(key, [n]);
  }
  for (const arr of index.values()) {
    arr.sort((a, b) => a.ctx_idx - b.ctx_idx);
  }

  let matched = 0;
  let exactCtxMatched = 0;
  const pinned: string[] = [];

  for (const entry of template) {
    const key = `${entry.layer}|${entry.feature}`;
    const candidates = index.get(key);
    if (!candidates || candidates.length === 0) continue;

    const desiredCtxIdx = targetLogitCtxIdx - entry.ctxDeltaFromLogit;
    let best = candidates[0];
    let bestDist = Math.abs(best.ctx_idx - desiredCtxIdx);
    for (let i = 1; i < candidates.length; i += 1) {
      const cand = candidates[i];
      const dist = Math.abs(cand.ctx_idx - desiredCtxIdx);
      if (dist < bestDist) {
        best = cand;
        bestDist = dist;
      }
      if (bestDist === 0) break;
    }

    matched += 1;
    if (best.ctx_idx === desiredCtxIdx) exactCtxMatched += 1;
    pinned.push(best.node_id);
  }

  const pinnedDedup = [...new Set(pinned)];
  return {
    pinnedIds: pinnedDedup,
    coverage: template.length === 0 ? 0 : matched / template.length,
    matched,
    exactCtxMatched,
    templateCount: template.length,
    targetLogitCtxIdx,
    targetToken,
    targetLogitProb: targetLogit.token_prob,
  };
}

async function postJsonWithRetry<T>(url: string, body: unknown, maxRetries: number, initialBackoffMs: number): Promise<T> {
  let attempt = 0;
  let backoff = initialBackoffMs;

  // eslint-disable-next-line no-constant-condition
  while (true) {
    attempt += 1;
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    const json = (await res.json()) as any;
    if (res.ok) return json as T;

    const errMsg = typeof json?.error === 'string' ? json.error : `HTTP ${res.status}`;
    const isBusy = errMsg.includes('RUNPOD_BUSY') || res.status === 503;
    if (!isBusy || attempt > maxRetries) {
      throw new Error(`Request failed (${attempt}/${maxRetries}): ${errMsg} ${json?.message ? `- ${json.message}` : ''}`);
    }

    await new Promise((r) => setTimeout(r, backoff));
    backoff = Math.min(backoff * 1.5, 30_000);
  }
}

async function main() {
  const args = parseArgs(process.argv);

  if (args.help) {
    console.log(`\nUsage (run from neuronpedia/apps/webapp):\n\n  npx tsx scripts/batch-eval-subgraph.ts \\\n+    --baseUrl http://localhost:3000 \\\n+    --modelId local-gemma-2b-toolformer \\\n+    --sourceSlug calculate35-1767789813119 \\\n+    --subgraphName test1 \\\n+    --prompts prompts.jsonl \\\n+    --out results.json \\\n+\nOptional:\n  --sourceSetName toolformer-fused\n  --maxNLogits 10\n  --desiredLogitProb 0.95\n  --nodeThreshold 0.8\n  --edgeThreshold 0.85\n  --maxFeatureNodes 5000\n  --maxRetries 10\n  --initialBackoffMs 2000\n`);
    return;
  }

  const baseUrl = requireString(args, 'baseUrl').replace(/\/$/, '');
  const modelId = requireString(args, 'modelId');
  const subgraphName = requireString(args, 'subgraphName');
  const promptsPath = requireString(args, 'prompts');
  const outPath = requireString(args, 'out');
  const sourceSlugArg = getOptionalString(args, 'sourceSlug');

  const sourceSetName = getOptionalString(args, 'sourceSetName');
  const maxNLogits = getNumberArg(args, 'maxNLogits', 10);
  const desiredLogitProb = getNumberArg(args, 'desiredLogitProb', 0.95);
  const nodeThreshold = getNumberArg(args, 'nodeThreshold', 0.8);
  const edgeThreshold = getNumberArg(args, 'edgeThreshold', 0.85);
  const maxFeatureNodes = getNumberArg(args, 'maxFeatureNodes', 5000);
  const maxRetries = getNumberArg(args, 'maxRetries', 10);
  const initialBackoffMs = getNumberArg(args, 'initialBackoffMs', 2000);

  const prisma = new PrismaClient();
  try {
    const prompts = await readJsonlPrompts(promptsPath);
    console.log(`Loaded prompts: ${prompts.length}`);

    let sourceSlugResolved: string;
    let sourceGraphMetadata: { id: string; url: string; slug: string };
    let subgraph: { id: string; displayName: string | null; pinnedIds: string[]; updatedAt: Date; userId: string };

    if (sourceSlugArg) {
      const meta = await prisma.graphMetadata.findUnique({
        where: { modelId_slug: { modelId, slug: sourceSlugArg } },
        select: { id: true, url: true, slug: true },
      });
      if (!meta) {
        throw new Error(`Source graph metadata not found for modelId=${modelId} slug=${sourceSlugArg}`);
      }

      const sg = await prisma.graphMetadataSubgraph.findFirst({
        where: {
          graphMetadataId: meta.id,
          displayName: subgraphName,
        },
        orderBy: { updatedAt: 'desc' },
        select: { id: true, displayName: true, pinnedIds: true, updatedAt: true, userId: true },
      });
      if (!sg) {
        throw new Error(`Subgraph not found: displayName=${subgraphName} on ${modelId}/${sourceSlugArg}`);
      }

      sourceSlugResolved = meta.slug;
      sourceGraphMetadata = { id: meta.id, url: meta.url, slug: meta.slug };
      subgraph = sg;
    } else {
      const sg = await prisma.graphMetadataSubgraph.findFirst({
        where: {
          displayName: subgraphName,
          graphMetadata: {
            modelId,
          },
        },
        orderBy: { updatedAt: 'desc' },
        select: {
          id: true,
          displayName: true,
          pinnedIds: true,
          updatedAt: true,
          userId: true,
          graphMetadata: { select: { id: true, slug: true, url: true } },
        },
      });
      if (!sg) {
        throw new Error(`Subgraph not found: displayName=${subgraphName} for modelId=${modelId}`);
      }

      sourceSlugResolved = sg.graphMetadata.slug;
      sourceGraphMetadata = { id: sg.graphMetadata.id, url: sg.graphMetadata.url, slug: sg.graphMetadata.slug };
      subgraph = {
        id: sg.id,
        displayName: sg.displayName,
        pinnedIds: sg.pinnedIds,
        updatedAt: sg.updatedAt,
        userId: sg.userId,
      };
    }

    const webappRoot = process.cwd();
    const sourceGraphUrl = sourceGraphMetadata.url?.trim();
    const sourceFileName = `${modelId}-${sourceSlugResolved}.json`;
    const sourceGraph = await (async () => {
      if (sourceGraphUrl && sourceGraphUrl.startsWith('/')) {
        return loadLocalGraphJsonFromWebappPublic(webappRoot, sourceGraphUrl);
      }
      return loadLocalGraphJsonFromWebappPublic(webappRoot, path.join('local-graphs', sourceFileName));
    })();

    const { template, sourceLogitCtxIdx, targetToken: sourceTargetToken } = buildTemplateFromSourceGraph(
      sourceGraph,
      subgraph.pinnedIds,
    );
    console.log(
      `Loaded subgraph ${subgraph.id} (name=${subgraph.displayName ?? ''}, userId=${subgraph.userId}, updatedAt=${subgraph.updatedAt.toISOString()})`,
    );
    console.log(
      `Template entries: ${template.length} (source target token=${sourceTargetToken ?? 'unknown'}, source logit ctx_idx=${sourceLogitCtxIdx})`,
    );

    const results: any[] = [];
    for (let i = 0; i < prompts.length; i += 1) {
      const item = prompts[i];
      const { prompt: promptForGraph, added: promptNewlineAdded } = ensureTrailingNewline(item.prompt);

      const rowBase: Record<string, unknown> = {
        index: i,
        label: item.label,
        promptNewlineAdded,
      };

      try {
        const payload: Record<string, unknown> = {
          prompt: promptForGraph,
          modelId,
          maxNLogits,
          desiredLogitProb,
          nodeThreshold,
          edgeThreshold,
          maxFeatureNodes,
        };
        if (sourceSetName) payload.sourceSetName = sourceSetName;

        const gen = await postJsonWithRetry<GenerateGraphResponse>(
          `${baseUrl}/api/graph/generate`,
          payload,
          maxRetries,
          initialBackoffMs,
        );
        if (!gen.graph) {
          throw new Error(`Generate returned no graph (message=${gen.message ?? ''}, url=${gen.url ?? ''})`);
        }
        const graph = gen.graph;
        const slug = graph.metadata?.slug;

        const mapped = mapTemplateToPinnedIds(template, graph);
        const behavior = summarizeBehaviorFromGraph(graph, item);

        const gateToken = '<';
        const gateScores = computeGateScoresIfAvailable(graph, mapped.pinnedIds, gateToken);
        const signedMarginScores = computeSignedMarginScoresIfAvailable(
          graph,
          mapped.pinnedIds,
          gateToken,
          behavior.echoFirstToken,
        );

        const full = computeGraphScoresFromGraphData(graph, []);
        const sub = computeGraphScoresFromGraphData(graph, mapped.pinnedIds);

        results.push({
          ...rowBase,
          slug,
          url: gen.url,
          targetToken: mapped.targetToken,
          targetProb: mapped.targetLogitProb,
          topToken: behavior.topToken,
          topProb: behavior.topProb,
          gateToken,
          echoFirstToken: behavior.echoFirstToken,
          expectedBehavior: behavior.expectedBehavior,
          decisionByTop1: behavior.decisionByTop1,
          probAngleBracket: behavior.probAngleBracket,
          rankAngleBracket: behavior.rankAngleBracket,
          probEchoFirst: behavior.probEchoFirst,
          rankEchoFirst: behavior.rankEchoFirst,
          probMargin: behavior.probMargin,
          signedProbMargin: behavior.signedProbMargin,
          topNonAngleToken: behavior.topNonAngleToken,
          topNonAngleProb: behavior.topNonAngleProb,
          salientLogitCount: behavior.salientLogitCount,
          salientCumulativeProbability: behavior.salientCumulativeProbability,
          hasAngleBracketTarget: behavior.hasAngleBracketTarget,
          hasEchoFirstTarget: behavior.hasEchoFirstTarget,
          hasSignedMarginTarget: behavior.hasSignedMarginTarget,
          ...(gateScores ?? {
            replacement_full_gate: null,
            completeness_full_gate: null,
            replacement_subgraph_gate: null,
            completeness_subgraph_gate: null,
          }),
          ...(signedMarginScores ?? {
            replacement_full_signed_margin: null,
            completeness_full_signed_margin: null,
            replacement_subgraph_signed_margin: null,
            completeness_subgraph_signed_margin: null,
          }),
          replacement_full: full.replacementScore,
          completeness_full: full.completenessScore,
          replacement_subgraph: sub.replacementScore,
          completeness_subgraph: sub.completenessScore,
          templateCount: mapped.templateCount,
          matched: mapped.matched,
          exactCtxMatched: mapped.exactCtxMatched,
          coverage: mapped.coverage,
        });

        if ((i + 1) % 10 === 0) {
          console.log(`Progress ${i + 1}/${prompts.length}`);
        }
      } catch (e) {
        results.push({
          ...rowBase,
          error: (e as Error)?.message ?? String(e),
        });
        console.error(`Failed index=${i}:`, (e as Error)?.message ?? e);
      }
    }

    const outDir = path.dirname(outPath);
    await fs.mkdir(outDir, { recursive: true });
    await fs.writeFile(outPath, JSON.stringify({
      meta: {
        modelId,
        sourceSlug: sourceSlugResolved,
        subgraphName,
        subgraphId: subgraph.id,
        sourceTargetToken: sourceTargetToken ?? null,
        gateToken: '<',
        templateSize: template.length,
        generatedAt: new Date().toISOString(),
        scoringSemantics: 'score-core.ts (merge non-pinned CLT features into error nodes by layer|ctx_idx)',
        behaviorSemantics: 'probMargin is P("<") - P(echoFirst); signedProbMargin flips sign for echo-first prompts; signed-margin circuit scores require both tokens to be present in the traced logits',
      },
      results,
    }, null, 2));

    // Write a sibling CSV for easy slicing.
    const csvPath = outPath.replace(/\.json$/i, '') + '.csv';
    const header = [
      'index',
      'label',
      'slug',
      'targetToken',
      'targetProb',
      'topToken',
      'topProb',
      'gateToken',
      'echoFirstToken',
      'expectedBehavior',
      'decisionByTop1',
      'probAngleBracket',
      'rankAngleBracket',
      'probEchoFirst',
      'rankEchoFirst',
      'probMargin',
      'signedProbMargin',
      'topNonAngleToken',
      'topNonAngleProb',
      'salientLogitCount',
      'salientCumulativeProbability',
      'hasAngleBracketTarget',
      'hasEchoFirstTarget',
      'hasSignedMarginTarget',
      'replacement_full_gate',
      'completeness_full_gate',
      'replacement_subgraph_gate',
      'completeness_subgraph_gate',
      'replacement_full_signed_margin',
      'completeness_full_signed_margin',
      'replacement_subgraph_signed_margin',
      'completeness_subgraph_signed_margin',
      'replacement_full',
      'completeness_full',
      'replacement_subgraph',
      'completeness_subgraph',
      'templateCount',
      'matched',
      'exactCtxMatched',
      'coverage',
      'error',
    ];
    const lines = [header.join(',')];
    for (const r of results) {
      const row = header
        .map((k) => {
          const v = (r as any)[k];
          const s = v === undefined || v === null ? '' : String(v);
          // Basic CSV escaping
          return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
        })
        .join(',');
      lines.push(row);
    }
    await fs.writeFile(csvPath, lines.join('\n'));
    console.log(`Wrote ${outPath} and ${csvPath}`);
  } finally {
    await prisma.$disconnect();
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
