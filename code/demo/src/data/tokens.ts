// Real TinyLlama token examples for "Write a Python script to sort a list."
// The model generates: "Here's a Python script that sorts a list:\n..."

export const EXAMPLE_PROMPT = 'Write a Python script to sort a list.'

// Head proposals: illustrative but plausible for TinyLlama on this prompt
export const HEAD_PROPOSALS = [
  // Head 0: top-10 tokens at depth 1 (predicts token at position t+2)
  { token: "Here", prob: 0.31, rank: 0 },
  { token: "This", prob: 0.18, rank: 1 },
  { token: "The", prob: 0.12, rank: 2 },
  { token: "def", prob: 0.08, rank: 3 },
  { token: "import", prob: 0.06, rank: 4 },
  { token: "A", prob: 0.05, rank: 5 },
  { token: "Sure", prob: 0.04, rank: 6 },
  { token: "#", prob: 0.03, rank: 7 },
  { token: "Below", prob: 0.02, rank: 8 },
  { token: "here", prob: 0.01, rank: 9 },
] as const

export const HEAD_PROPOSALS_1 = [
  { token: "'s", prob: 0.55, rank: 0 },
  { token: " is", prob: 0.28, rank: 1 },
  { token: " are", prob: 0.17, rank: 2 },
] as const

export const HEAD_PROPOSALS_2 = [
  { token: "a", prob: 0.62, rank: 0 },
  { token: "an", prob: 0.38, rank: 1 },
] as const

export const HEAD_PROPOSALS_3 = [
  { token: "Python", prob: 0.71, rank: 0 },
  { token: "simple", prob: 0.29, rank: 1 },
] as const

// Tree node token assignments (what token each node represents in the demo)
// Index maps to MEDUSA_TREE_NODES
export const TREE_NODE_TOKENS: Record<number, string> = {
  // depth 1 root (1 node)
  0: "here",
  // depth 2 (3 nodes) — children of root
  1: "'s",     2: " is",    3: " are",
  // depth 3 (6 nodes)
  4: " a",     5: " the",   // children of node 1 "'s"
  6: " the",   7: " a",     // children of node 2 " is"
  8: " both",  9: " all",   // children of node 3 " are"
  // depth 4 (10 nodes)
  10: "Python", 11: "simple",             // children of node 4 " a"
  12: "great",  13: "good",               // children of node 5 " the"
  14: "quick",  15: "sorted",             // children of node 6
  16: "best",                             // children of node 7
  17: "easiest",18: "fastest",            // children of node 8
  19: "working",                          // children of node 9
}

// Demo: accepted path for "Write a Python script to sort a list."
// Path spells: "here" → "'s" → " a" → "Python" = "here's a Python [script...]"
export const GREEDY_ACCEPTED_PATH = [0, 1, 4, 10]
export const TYPICAL_ACCEPTED_PATH = [0, 1, 4, 10]  // same for this example
// The "borderline" node that typical accepts but greedy might reject in marginal cases
export const BORDERLINE_NODE = 5  // " the" — lower probability path

// Generated token stream for hero section animation
export const AUTOREGRESSIVE_TOKENS = [
  "Here", "'s", " a", " Python", " script", " that", " sorts", " a",
  " list", ":", "\n\n", "```", "python", "\n", "#", " Example"
]

export const MEDUSA_TOKEN_GROUPS = [
  ["Here", "'s", " a"],
  [" Python", " script", " that"],
  [" sorts", " a", " list"],
  [":", "\n\n", "```"],
  ["python", "\n", "#"],
]
