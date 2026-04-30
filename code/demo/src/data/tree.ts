export const MEDUSA_TREE_NODES: ReadonlyArray<ReadonlyArray<number>> = [
  // depth 1 — 1 root
  [0],
  // depth 2 — 3 nodes
  [0,0],[0,1],[0,2],
  // depth 3 — 6 nodes
  [0,0,0],[0,0,1],[0,1,0],[0,1,1],[0,2,0],[0,2,1],
  // depth 4 — 10 nodes
  [0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],
  [0,1,0,0],[0,1,0,1],[0,1,1,0],[0,2,0,0],[0,2,0,1],[0,2,1,0],
]

export const MEDUSA_TREE_PARENT_INDICES: ReadonlyArray<number> = [
  -1,
  0,0,0,
  1,1,2,2,3,3,
  4,4,5,5,6,6,7,8,8,9,
]

export type TreeNode = {
  id: number
  path: ReadonlyArray<number>
  depth: number       // 1–4
  parentId: number    // -1 if root
  childIds: number[]
}

export function buildTreeNodes(): TreeNode[] {
  const parents = MEDUSA_TREE_PARENT_INDICES as number[]
  const nodes: TreeNode[] = MEDUSA_TREE_NODES.map((path, id) => ({
    id,
    path,
    depth: path.length,
    parentId: parents[id],
    childIds: [],
  }))
  nodes.forEach(node => {
    if (node.parentId >= 0) {
      nodes[node.parentId].childIds.push(node.id)
    }
  })
  return nodes
}

export function getDepthColor(depth: number): string {
  switch (depth) {
    case 1: return '#f59e0b'  // amber-500
    case 2: return '#10b981'  // emerald-500
    case 3: return '#0ea5e9'  // sky-500
    case 4: return '#8b5cf6'  // violet-500
    default: return '#94a3b8'
  }
}

export function getDepthBg(depth: number): string {
  switch (depth) {
    case 1: return 'bg-amber-100 text-amber-800 border-amber-300'
    case 2: return 'bg-emerald-100 text-emerald-800 border-emerald-300'
    case 3: return 'bg-sky-100 text-sky-800 border-sky-300'
    case 4: return 'bg-violet-100 text-violet-800 border-violet-300'
    default: return 'bg-slate-100 text-slate-800 border-slate-300'
  }
}

export function computeAttentionMatrix(
  parentIndices: number[],
  prefixLen: number
): boolean[][] {
  const n = parentIndices.length
  const totalCols = prefixLen + n
  const mask = Array.from({ length: n }, () => new Array(totalCols).fill(false) as boolean[])

  for (let i = 0; i < n; i++) {
    for (let c = 0; c < prefixLen; c++) mask[i][c] = true
    let curr: number = i
    while (curr !== -1) {
      mask[i][prefixLen + curr] = true
      curr = parentIndices[curr]
    }
  }
  return mask
}

export function getAncestorIds(nodeId: number, parentIndices: number[]): number[] {
  const ancestors: number[] = []
  let curr = parentIndices[nodeId]
  while (curr !== -1) {
    ancestors.push(curr)
    curr = parentIndices[curr]
  }
  return ancestors
}

export const DEPTH_COUNTS = [1, 3, 6, 10]
export const S_K = [10, 6, 4, 3]
export const TREE_NODES = buildTreeNodes()
