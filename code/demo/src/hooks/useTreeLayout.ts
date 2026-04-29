import { useMemo } from 'react'
import * as d3 from 'd3'
import type { TreeNode } from '../data/tree'

type LayoutDatum = {
  id: string
  parentId: string | null
  data: TreeNode | null
}

export function useTreeLayout(
  nodes: TreeNode[],
  width: number,
  height: number
) {
  return useMemo(() => {
    const allDatums: LayoutDatum[] = [
      { id: 'virtual-root', parentId: null, data: null },
      ...nodes.map(n => ({
        id: String(n.id),
        parentId: n.parentId === -1 ? 'virtual-root' : String(n.parentId),
        data: n,
      })),
    ]

    const root = d3
      .stratify<LayoutDatum>()
      .id(d => d.id)
      .parentId(d => d.parentId)(allDatums)

    const layout = d3
      .tree<LayoutDatum>()
      .size([width, height])
      .separation((a, b) => {
        if (a.parent === b.parent) return 1.4
        return 2.5
      })

    layout(root)
    return root
  }, [nodes, width, height])
}
