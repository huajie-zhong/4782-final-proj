import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import * as d3 from 'd3'
import { TREE_NODES, getDepthColor, MEDUSA_TREE_PARENT_INDICES } from '../../data/tree'
import { TREE_NODE_TOKENS, GREEDY_ACCEPTED_PATH } from '../../data/tokens'
import { useTreeLayout } from '../../hooks/useTreeLayout'

const SVG_W = 900
const SVG_H = 320

type NodeState = 'pending' | 'checking' | 'accepted' | 'rejected'

type Props = { inView: boolean; animStep: number }

export default function AcceptanceWalkViz({ inView, animStep }: Props) {
  const root = useTreeLayout(TREE_NODES, SVG_W - 60, SVG_H - 60)
  const [nodeStates, setNodeStates] = useState<Record<number, NodeState>>({})

  const acceptedPath = GREEDY_ACCEPTED_PATH

  useEffect(() => {
    if (!inView) {
      setNodeStates({})
      return
    }
    const states: Record<number, NodeState> = {}

    // Step-by-step walk following GREEDY_ACCEPTED_PATH: 0 -> 1 -> 4 -> 10
    if (animStep >= 1) states[0] = 'checking'
    if (animStep >= 2) states[0] = 'accepted'

    if (animStep >= 3) {
      states[1] = 'checking'
      states[2] = 'pending'
      states[3] = 'pending'
    }
    if (animStep >= 4) {
      states[1] = 'accepted'
      states[2] = 'rejected'
      states[3] = 'rejected'
      // Descendants of 2, 3
      for (let i = 6; i <= 9; i++) states[i] = 'rejected'
      for (let i = 14; i <= 19; i++) states[i] = 'rejected'
    }

    if (animStep >= 5) {
      states[4] = 'checking'
      states[5] = 'pending'
    }
    if (animStep >= 6) {
      states[4] = 'accepted'
      states[5] = 'rejected'
      // Descendants of 5
      states[12] = 'rejected'
      states[13] = 'rejected'
    }

    if (animStep >= 7) {
      states[10] = 'checking'
      states[11] = 'pending'
    }
    if (animStep >= 8) {
      states[10] = 'accepted'
      states[11] = 'rejected'
    }

    setNodeStates(states)
  }, [animStep, inView])

  const nodeDescendants = root.descendants().filter(d => d.data.data !== null)
  const links = root.links().filter(l => l.source.data.data !== null)

  const getNodeFill = (id: number): string => {
    const state = nodeStates[id]
    if (state === 'accepted') return '#10b981'
    if (state === 'rejected') return '#ef4444'
    if (state === 'checking') return '#f59e0b'
    return '#94a3b8'
  }

  const getNodeOpacity = (id: number): number => {
    const state = nodeStates[id]
    if (state === 'rejected') return 0.25
    if (!state) return 0.55
    return 1
  }

  const acceptedTokens = acceptedPath
    .filter(i => nodeStates[i] === 'accepted')
    .map(i => TREE_NODE_TOKENS[i])

  return (
    <div className="flex flex-col gap-4">
      {/* Accepted tokens display */}
      <div className="bg-white border border-birch rounded p-4">
        <div className="text-xs font-mono text-ink-soft mb-2">Accepted tokens this pass:</div>
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-mono text-sm text-ink-soft px-2 py-1 bg-parchment rounded border border-birch">
            "Write a Python script..."
          </span>
          <span className="text-birch">→</span>
          {acceptedTokens.length > 0 ? (
            acceptedTokens.map((tok, i) => (
              <motion.span
                key={i}
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                className="font-mono text-sm px-2 py-1 bg-grove-light text-grove rounded border border-grove"
              >
                {tok}
              </motion.span>
            ))
          ) : (
            <span className="text-birch text-sm font-mono">watching...</span>
          )}
          {acceptedTokens.length === 4 && (
            <motion.span
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="ml-2 text-sm font-mono text-grove"
            >
              4 tokens accepted in one pass
            </motion.span>
          )}
        </div>
      </div>

      {/* Tree */}
      <div className="overflow-x-auto border border-birch rounded bg-white">
        <svg width={SVG_W} height={SVG_H + 20} style={{ display: 'block', minWidth: SVG_W }}>
          <g transform="translate(30, 30)">
            {links.map((link, i) => {
              const src = link.source as d3.HierarchyPointNode<typeof link.source.data>
              const tgt = link.target as d3.HierarchyPointNode<typeof link.target.data>
              const srcId = link.source.data.data?.id
              const tgtId = link.target.data.data?.id
              const isOnPath = srcId !== undefined && tgtId !== undefined &&
                acceptedPath.includes(srcId) && acceptedPath.includes(tgtId) &&
                nodeStates[tgtId] === 'accepted'
              return (
                <line
                  key={i}
                  x1={src.x ?? 0} y1={src.y ?? 0}
                  x2={tgt.x ?? 0} y2={tgt.y ?? 0}
                  stroke={isOnPath ? '#10b981' : '#cbd5e1'}
                  strokeWidth={isOnPath ? 2.5 : 0.8}
                  strokeOpacity={isOnPath ? 0.9 : 0.4}
                />
              )
            })}

            {nodeDescendants.map(node => {
              const nd = node.data.data!
              const fill = getNodeFill(nd.id)
              const opacity = getNodeOpacity(nd.id)
              const isChecking = nodeStates[nd.id] === 'checking'
              const isAccepted = nodeStates[nd.id] === 'accepted'
              return (
                <g key={nd.id} opacity={opacity}>
                  {isChecking && (
                    <motion.circle
                      cx={node.x ?? 0} cy={node.y ?? 0}
                      r={14}
                      fill="#f59e0b"
                      opacity={0.3}
                      animate={{ r: [10, 18, 10], opacity: [0.4, 0.1, 0.4] }}
                      transition={{ repeat: Infinity, duration: 0.7 }}
                    />
                  )}
                  <motion.circle
                    cx={node.x ?? 0}
                    cy={node.y ?? 0}
                    r={nd.depth === 1 ? 7 : 5}
                    fill={fill}
                    stroke="white"
                    strokeWidth={1.5}
                    animate={{
                      r: isChecking ? [7, 9, 7] : (nd.depth === 1 ? 7 : 5),
                    }}
                    transition={{ repeat: isChecking ? Infinity : 0, duration: 0.5 }}
                    style={{ transformOrigin: `${node.x ?? 0}px ${node.y ?? 0}px` }}
                  />
                  {nd.depth === 1 && (
                    <text
                      x={node.x ?? 0}
                      y={(node.y ?? 0) - 12}
                      textAnchor="middle"
                      fill={fill}
                      fontSize={8}
                      fontFamily="JetBrains Mono, monospace"
                      opacity={opacity}
                    >
                      {TREE_NODE_TOKENS[nd.id]}
                    </text>
                  )}
                  {isAccepted && nd.depth > 1 && (
                    <text
                      x={node.x ?? 0}
                      y={(node.y ?? 0) - 10}
                      textAnchor="middle"
                      fill="#10b981"
                      fontSize={8}
                      fontFamily="JetBrains Mono, monospace"
                    >
                      {TREE_NODE_TOKENS[nd.id]}
                    </text>
                  )}
                </g>
              )
            })}
          </g>
        </svg>
      </div>

      {/* Legend */}
      <div className="flex gap-4 text-xs flex-wrap text-ink-soft font-mono">
        {[
          { color: '#94a3b8', label: 'pending' },
          { color: '#f59e0b', label: 'checking' },
          { color: '#10b981', label: 'accepted' },
          { color: '#ef4444', label: 'rejected' },
        ].map(({ color, label }) => (
          <span key={label} className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-sm inline-block" style={{ backgroundColor: color }} />
            {label}
          </span>
        ))}
      </div>
    </div>
  )
}
