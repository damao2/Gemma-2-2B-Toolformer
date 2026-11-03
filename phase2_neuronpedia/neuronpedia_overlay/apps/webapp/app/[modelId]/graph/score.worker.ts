/* eslint-disable no-restricted-globals */
// Lightweight worker to compute subgraph scores off the main thread.
// Delegates to shared score-core to keep semantics identical with offline scripts.

import { computeGraphScoresFromGraphData, ScoreGraph } from './score-core';

console.log('Worker script loaded');

self.onmessage = (ev: MessageEvent) => {
  console.log('Worker received message:', ev.data ? 'has data' : 'null data');

  try {
    // Handle Chrome's stricter Web Worker message handling
    if (!ev.data) {
      // Chrome may send null data during worker initialization - ignore these messages
      return;
    }

    const { data } = ev;

    // Validate that we have the expected message structure
    // eslint-disable-next-line no-prototype-builtins
    if (typeof data !== 'object' || !data.hasOwnProperty('graph') || !data.hasOwnProperty('requestId')) {
      console.error('Worker: Invalid message format', data);
      // @ts-ignore
      self.postMessage({ error: 'Invalid message format received by worker' });
      return;
    }

    console.log('Worker: starting computation');
    const { requestId, graph, pinnedIds } = data as { requestId: number; graph: ScoreGraph; pinnedIds: string[] };
    const scores = computeGraphScoresFromGraphData(graph, pinnedIds);
    console.log('Worker: computation complete, sending results:', scores);
    // @ts-ignore
    self.postMessage({ requestId, ...scores });
  } catch (err) {
    console.error('Worker error:', err);
    // @ts-ignore
    self.postMessage({ error: (err as Error)?.message || 'Unknown worker error' });
  }
};
