// Shared, dependency-free scoring logic for attribution graphs.
// Used by both the UI WebWorker and offline/batch scripts.

import { CLTGraphLink, CLTGraphNode } from './graph-types';

export type ScoreGraph = {
  nodes: CLTGraphNode[];
  links: CLTGraphLink[];
};

export type GraphScoreResult = {
  replacementScore: number;
  completenessScore: number;
};

type GraphScoreOptions = {
  useAbsoluteInfluence?: boolean;
};

function normalizeMatrix(matrix: number[][]): number[][] {
  return matrix.map((row) => {
    const absRow = row.map((val) => Math.abs(val));
    const sum = absRow.reduce((acc, val) => acc + val, 0);
    const clampedSum = Math.max(sum, 1e-10);
    return absRow.map((val) => val / clampedSum);
  });
}

function computeInfluence(A: number[][], logitWeights: number[], maxIter: number = 1000): number[] {
  let currentInfluence = new Array(A[0].length).fill(0);
  for (let j = 0; j < A[0].length; j += 1) {
    for (let i = 0; i < logitWeights.length; i += 1) {
      currentInfluence[j] += logitWeights[i] * A[i][j];
    }
  }

  const influence = [...currentInfluence];
  let iterations = 0;

  while (currentInfluence.some((val) => val !== 0)) {
    if (iterations >= maxIter) {
      throw new Error(`Influence computation failed to converge after ${iterations} iterations`);
    }

    const newInfluence = new Array(A[0].length).fill(0);
    for (let j = 0; j < A[0].length; j += 1) {
      for (let i = 0; i < currentInfluence.length; i += 1) {
        newInfluence[j] += currentInfluence[i] * A[i][j];
      }
    }

    currentInfluence = newInfluence;
    for (let i = 0; i < influence.length; i += 1) {
      influence[i] += currentInfluence[i];
    }
    iterations += 1;
  }

  return influence;
}

function reconstructAdjacencyMatrix(
  nodes: CLTGraphNode[],
  edges: CLTGraphLink[],
): { matrix: number[][]; sortedNodes: CLTGraphNode[] } {
  const featureTypeOrder = ['cross layer transcoder', 'mlp reconstruction error', 'embedding', 'logit'];

  function getSortKey(node: CLTGraphNode) {
    let typePriority = featureTypeOrder.indexOf(node.feature_type);
    if (typePriority === -1) typePriority = featureTypeOrder.length;
    const layerNum =
      node.layer === 'E' ? 0 : Number.isNaN(parseInt(node.layer, 10)) ? 999 : parseInt(node.layer, 10);
    return [typePriority, layerNum, node.ctx_idx, node.feature || 0];
  }

  const sortedNodes = [...nodes].sort((a, b) => {
    const keyA = getSortKey(a);
    const keyB = getSortKey(b);
    for (let i = 0; i < Math.max(keyA.length, keyB.length); i += 1) {
      const valA = (keyA as number[])[i] || 0;
      const valB = (keyB as number[])[i] || 0;
      if (valA !== valB) return valA - valB;
    }
    return 0;
  });

  const nodeIdToIdx: Record<string, number> = {};
  sortedNodes.forEach((node, idx) => {
    nodeIdToIdx[node.node_id] = idx;
  });

  const nNodes = sortedNodes.length;
  const adjacencyMatrix: number[][] = Array(nNodes)
    .fill(null)
    .map(() => Array(nNodes).fill(0));

  edges.forEach((edge) => {
    const srcIdx = nodeIdToIdx[edge.source];
    const dstIdx = nodeIdToIdx[edge.target];
    if (srcIdx !== undefined && dstIdx !== undefined) {
      adjacencyMatrix[dstIdx][srcIdx] = edge.weight;
    }
  });

  return { matrix: adjacencyMatrix, sortedNodes };
}

function finalizeGraphScores(
  adjacencyMatrix: number[][],
  sortedNodes: CLTGraphNode[],
  logitWeights: number[],
  options: GraphScoreOptions = {},
): GraphScoreResult {
  const nFeatures = sortedNodes.filter((n) => n.feature_type === 'cross layer transcoder').length;
  const nErrorNodes = sortedNodes.filter((n) => n.feature_type === 'mlp reconstruction error').length;
  const nTokens = sortedNodes.filter((n) => n.feature_type === 'embedding').length;

  const errorStart = nFeatures;
  const errorEnd = errorStart + nErrorNodes;
  const tokenEnd = errorEnd + nTokens;

  const normalizedMatrix = normalizeMatrix(adjacencyMatrix);
  const nodeInfluence = computeInfluence(normalizedMatrix, logitWeights);
  const useAbsoluteInfluence = options.useAbsoluteInfluence ?? false;

  const tokenInfluence = (useAbsoluteInfluence
    ? nodeInfluence.slice(errorEnd, tokenEnd).map((val) => Math.abs(val))
    : nodeInfluence.slice(errorEnd, tokenEnd)
  ).reduce((sum, val) => sum + val, 0);
  const errorInfluence = (useAbsoluteInfluence
    ? nodeInfluence.slice(errorStart, errorEnd).map((val) => Math.abs(val))
    : nodeInfluence.slice(errorStart, errorEnd)
  ).reduce((sum, val) => sum + val, 0);
  const replacementDenominator = tokenInfluence + errorInfluence;
  const replacementScore = replacementDenominator === 0 ? 0 : tokenInfluence / replacementDenominator;

  const nonErrorFractions = normalizedMatrix.map(
    (row) => 1 - row.slice(errorStart, errorEnd).reduce((sum, val) => sum + val, 0),
  );
  const outputInfluence = nodeInfluence.map((val, i) => val + logitWeights[i]);
  const weightedOutputInfluence = useAbsoluteInfluence ? outputInfluence.map((val) => Math.abs(val)) : outputInfluence;
  const outputInfluenceSum = weightedOutputInfluence.reduce((sum, val) => sum + val, 0);
  const completenessScore =
    nonErrorFractions.map((fraction, i) => fraction * weightedOutputInfluence[i]).reduce((sum, val) => sum + val, 0) /
    (outputInfluenceSum === 0 ? 1 : outputInfluenceSum);

  return {
    replacementScore: Number.isNaN(replacementScore) ? 0 : replacementScore,
    completenessScore: Number.isNaN(completenessScore) ? 0 : completenessScore,
  };
}

export function computeGraphScoresFromGraphData(
  graphData: ScoreGraph,
  pinnedIds: string[] = [],
): GraphScoreResult {
  const graphNodesToUse = graphData.nodes;

  const { matrix: adjacencyMatrix, sortedNodes } = reconstructAdjacencyMatrix(graphNodesToUse, graphData.links);

  const zeroRowAndColumn = (idx: number) => {
    for (let c = 0; c < adjacencyMatrix.length; c += 1) {
      adjacencyMatrix[idx][c] = 0;
    }
    for (let r = 0; r < adjacencyMatrix.length; r += 1) {
      adjacencyMatrix[r][idx] = 0;
    }
  };

  // If pinnedIds provided, merge non-pinned feature nodes into their corresponding error nodes
  if (pinnedIds.length > 0) {
    const pinnedSet = new Set(pinnedIds);

    const errorIndexByKey: Record<string, number> = {};
    sortedNodes.forEach((node, idx) => {
      if (node.feature_type === 'mlp reconstruction error') {
        const key = `${node.layer}|${node.ctx_idx}`;
        errorIndexByKey[key] = idx;
      }
    });

    sortedNodes.forEach((node, featureIdx) => {
      if (node.feature_type !== 'cross layer transcoder') return;
      if (pinnedSet.has(node.node_id)) return;

      const key = `${node.layer}|${node.ctx_idx}`;
      const errorIdx = errorIndexByKey[key];
      if (errorIdx === undefined) {
        zeroRowAndColumn(featureIdx);
        return;
      }

      for (let r = 0; r < adjacencyMatrix.length; r += 1) {
        adjacencyMatrix[r][errorIdx] += adjacencyMatrix[r][featureIdx];
      }

      zeroRowAndColumn(featureIdx);
    });
  }

  const nLogits = sortedNodes.filter((n) => n.feature_type === 'logit').length;

  const logitWeights = new Array(adjacencyMatrix.length).fill(0);
  const sortedLogitProbs = sortedNodes
    .filter((n) => n.feature_type === 'logit')
    .map((n) => n.token_prob);
  for (let i = 0; i < nLogits; i += 1) {
    logitWeights[adjacencyMatrix.length - nLogits + i] = sortedLogitProbs[i];
  }

  return finalizeGraphScores(adjacencyMatrix, sortedNodes, logitWeights);
}

export function computeGraphScoresFromGraphDataWithLogitWeightOverride(
  graphData: ScoreGraph,
  pinnedIds: string[] = [],
  getLogitWeight: (node: CLTGraphNode) => number,
  options: GraphScoreOptions = {},
): GraphScoreResult {
  const graphNodesToUse = graphData.nodes;

  const { matrix: adjacencyMatrix, sortedNodes } = reconstructAdjacencyMatrix(graphNodesToUse, graphData.links);

  const zeroRowAndColumn = (idx: number) => {
    for (let c = 0; c < adjacencyMatrix.length; c += 1) {
      adjacencyMatrix[idx][c] = 0;
    }
    for (let r = 0; r < adjacencyMatrix.length; r += 1) {
      adjacencyMatrix[r][idx] = 0;
    }
  };

  if (pinnedIds.length > 0) {
    const pinnedSet = new Set(pinnedIds);

    const errorIndexByKey: Record<string, number> = {};
    sortedNodes.forEach((node, idx) => {
      if (node.feature_type === 'mlp reconstruction error') {
        const key = `${node.layer}|${node.ctx_idx}`;
        errorIndexByKey[key] = idx;
      }
    });

    sortedNodes.forEach((node, featureIdx) => {
      if (node.feature_type !== 'cross layer transcoder') return;
      if (pinnedSet.has(node.node_id)) return;

      const key = `${node.layer}|${node.ctx_idx}`;
      const errorIdx = errorIndexByKey[key];
      if (errorIdx === undefined) {
        zeroRowAndColumn(featureIdx);
        return;
      }

      for (let r = 0; r < adjacencyMatrix.length; r += 1) {
        adjacencyMatrix[r][errorIdx] += adjacencyMatrix[r][featureIdx];
      }

      zeroRowAndColumn(featureIdx);
    });
  }

  const logitWeights = new Array(adjacencyMatrix.length).fill(0);
  for (let i = 0; i < sortedNodes.length; i += 1) {
    const node = sortedNodes[i];
    if (node.feature_type !== 'logit') continue;
    const w = getLogitWeight(node);
    logitWeights[i] = Number.isFinite(w) ? w : 0;
  }

  // Note: if the override weights are all 0, these scores are not meaningful.
  // Callers should guard and treat that case as "missing".
  return finalizeGraphScores(adjacencyMatrix, sortedNodes, logitWeights, options);
}

export function computeGraphScoresFromGraphDataWithSignedLogitWeightOverride(
  graphData: ScoreGraph,
  pinnedIds: string[] = [],
  getLogitWeight: (node: CLTGraphNode) => number,
): GraphScoreResult {
  return computeGraphScoresFromGraphDataWithLogitWeightOverride(graphData, pinnedIds, getLogitWeight, {
    useAbsoluteInfluence: true,
  });
}
