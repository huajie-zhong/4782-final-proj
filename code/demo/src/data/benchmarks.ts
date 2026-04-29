// Real numbers from final/results/TinyLlama-1.1B-Chat-v1.0_benchmark.json
// and final/results/table3_TinyLlama-1.1B-Chat-v1.0.json

export const BENCHMARK = {
  model: 'TinyLlama-1.1B-Chat-v1.0',
  greedyTps: 33.39,
  medusaTps: 67.16,
  speedupRatio: 2.011,
  avgAcceptanceRate: 1.540,
  medusaTypicalTps: 72.07,
  speedupRatioTypical: 2.158,
  avgAcceptanceRateTypical: 1.923,
  headAccuracies: {
    head0: 0.5206,
    head1: 0.3774,
    head2: 0.2885,
    head3: 0.2367,
  },
  perPromptGreedy: [
    { prompt: 'Write a Python script to sort a list.', acceptanceRate: 1.909, tps: 76.51 },
    { prompt: 'Explain the theory of relativity in simple terms.', acceptanceRate: 1.228, tps: 58.90 },
    { prompt: 'What are the benefits of eating healthy?', acceptanceRate: 1.867, tps: 76.00 },
    { prompt: 'Compose a short poem about the moon.', acceptanceRate: 1.115, tps: 56.25 },
    { prompt: 'Give me a recipe for chocolate chip cookies.', acceptanceRate: 1.580, tps: 68.12 },
  ],
  perPromptTypical: [
    { prompt: 'Write a Python script to sort a list.', acceptanceRate: 2.000, tps: 73.05 },
    { prompt: 'Explain the theory of relativity in simple terms.', acceptanceRate: 1.745, tps: 67.89 },
    { prompt: 'What are the benefits of eating healthy?', acceptanceRate: 2.071, tps: 74.88 },
    { prompt: 'Compose a short poem about the moon.', acceptanceRate: 2.098, tps: 76.80 },
    { prompt: 'Give me a recipe for chocolate chip cookies.', acceptanceRate: 1.702, tps: 67.75 },
  ],
}

// From table3_TinyLlama-1.1B-Chat-v1.0.json
export const TABLE3 = {
  greedyTps: 33.64,
  rows: {
    none: { medusaTps: 53.00, speedup: 1.576, avgAcceptanceRate: 0.867 },
    naive: { medusaTps: 49.07, speedup: 1.459, avgAcceptanceRate: 1.559 },
    optimized: { medusaTps: 64.88, speedup: 1.929, avgAcceptanceRate: 1.540 },
  },
}

export const ABLATION_DATA = [
  { name: 'Autoregressive\n(baseline)', speedup: 1.0, fill: '#94a3b8' },
  { name: 'Heads only\n(no tree)', speedup: 1.576, fill: '#f59e0b' },
  { name: 'Naive\nCartesian tree', speedup: 1.459, fill: '#f97316' },
  { name: 'Optimized\ntree (64 nodes)', speedup: 1.929, fill: '#10b981' },
  { name: 'Optimized +\ntypical accept', speedup: 2.158, fill: '#8b5cf6' },
]

export const HEAD_ACCURACY_DATA = [
  { head: 'Head 0\n(+2 steps)', accuracy: 52.1 },
  { head: 'Head 1\n(+3 steps)', accuracy: 37.7 },
  { head: 'Head 2\n(+4 steps)', accuracy: 28.9 },
  { head: 'Head 3\n(+5 steps)', accuracy: 23.7 },
]
