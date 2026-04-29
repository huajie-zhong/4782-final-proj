import { useState } from 'react'
import { motion } from 'framer-motion'
import * as d3 from 'd3'
import { TREE_NODES, getDepthColor, DEPTH_COUNTS } from '../../data/tree'
import { TREE_NODE_TOKENS } from '../../data/tokens'
import { useTreeLayout } from '../../hooks/useTreeLayout'

const SVG_W = 800
const SVG_H = 380

type Props = { inView: boolean }

type SelectedNode = { id: number; x: number; y: number } | null

function getAncestorIds(nodeId: number): number[] {
  const ids: number[] = []
  let current = TREE_NODES[nodeId]
  if (!current) return ids
  let parentId = current.parentId
  while (parentId !== -1) {
    ids.push(parentId)
    current = TREE_NODES[parentId]
    parentId = current.parentId
  }
  return ids
}

export default function CandidateTreeViz({ inView }: Props) {
  const [selected, setSelected] = useState<SelectedNode>(null)
  const root = useTreeLayout(TREE_NODES, SVG_W - 60, SVG_H - 80)

  const links = root.links().filter(l => l.source.data.data !== null)
  const nodeDescendants = root.descendants().filter(d => d.data.data !== null)

  const selectedAncestors = selected ? new Set(getAncestorIds(selected.id)) : new Set<number>()
  const isOnPath = (id: number) => selected !== null && (id === selected.id || selectedAncestors.has(id))

  return (
    <div className="w-full">
      {/* Depth legend */}
      <div className="flex gap-4 mb-3 flex-wrap">
        {[1, 2, 3, 4].map(d => (
          <div key={d} className="flex items-center gap-1.5 text-xs font-mono text-ink-soft">
            <div className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: getDepthColor(d) }} />
            depth {d} — {DEPTH_COUNTS[d - 1]} nodes
          </div>
        ))}
        {selected && (
          <button
            onClick={() => setSelected(null)}
            className="ml-auto text-xs font-mono text-ink-soft hover:text-grove transition-colors"
          >
            clear selection ×
          </button>
        )}
      </div>

      {/* Scrollable tree SVG */}
      <div className="overflow-x-auto border border-birch rounded bg-white">
        <svg
          width={SVG_W}
          height={SVG_H}
          style={{ display: 'block', minWidth: SVG_W }}
        >
          <g transform="translate(30, 40)">
            {/* Links */}
            {links.map((link, i) => {
              const src = link.source as d3.HierarchyPointNode<typeof link.source.data>
              const tgt = link.target as d3.HierarchyPointNode<typeof link.target.data>
              const depth = (link.target.data.data?.depth ?? 1)
              const srcId = link.source.data.data?.id
              const tgtId = link.target.data.data?.id
              const highlighted = selected && srcId !== undefined && tgtId !== undefined &&
                isOnPath(srcId) && isOnPath(tgtId)
              return (
                <motion.line
                  key={i}
                  x1={src.x ?? 0} y1={src.y ?? 0}
                  x2={tgt.x ?? 0} y2={tgt.y ?? 0}
                  stroke={highlighted ? getDepthColor(depth) : getDepthColor(depth)}
                  strokeOpacity={selected ? (highlighted ? 0.9 : 0.08) : 0.35}
                  strokeWidth={highlighted ? 2.5 : 1.2}
                  initial={{ opacity: 0 }}
                  animate={inView ? { opacity: 1 } : { opacity: 0 }}
                  transition={{ delay: depth * 0.15 + i * 0.003, duration: 0.4 }}
                />
              )
            })}

            {/* Nodes */}
            {nodeDescendants.map(node => {
              const nd = node.data.data!
              const color = getDepthColor(nd.depth)
              const token = TREE_NODE_TOKENS[nd.id]
              const isSelected = selected?.id === nd.id
              const isAncestor = selectedAncestors.has(nd.id)
              const dimmed = selected && !isSelected && !isAncestor

              return (
                <g key={nd.id}>
                  {/* Interactivity hint: complex tap target */}
                  {!selected && nd.depth === 1 && (
                    <motion.g
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: 1.5 }}
                    >
                      {/* Outer rotating dashed ring */}
                      <motion.circle
                        cx={node.x}
                        cy={node.y}
                        r={14}
                        fill="none"
                        stroke={color}
                        strokeWidth={1}
                        strokeDasharray="2,2"
                        animate={{ rotate: 360 }}
                        transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
                        style={{ pointerEvents: 'none' }}
                      />
                      {/* Inner pulsing solid ring */}
                      <motion.circle
                        cx={node.x}
                        cy={node.y}
                        r={12}
                        fill={color}
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ 
                          opacity: [0, 0.4, 0],
                          scale: [0.8, 2.2],
                        }}
                        transition={{ 
                          duration: 1.8, 
                          repeat: Infinity, 
                          ease: "easeOut" 
                        }}
                        style={{ pointerEvents: 'none' }}
                      />
                    </motion.g>
                  )}
                  
                  {/* Focus ring for selected node (replaces boldness) */}
                  {isSelected && (
                    <motion.circle
                      cx={node.x}
                      cy={node.y}
                      r={nd.depth === 1 ? 14 : 10}
                      fill="none"
                      stroke={color}
                      strokeWidth={1.5}
                      initial={{ scale: 0, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      transition={{ type: 'spring', stiffness: 300, damping: 25 }}
                    />
                  )}

                  <motion.circle
                    cx={node.x}
                    cy={node.y}
                    r={nd.depth === 1 ? 8 : 5}
                    fill={color}
                    fillOpacity={dimmed ? 0.12 : isSelected ? 1 : isAncestor ? 0.8 : 0.85}
                    stroke="white"
                    strokeWidth={1.5}
                    filter="drop-shadow(0 2px 3px rgba(0,0,0,0.1))"
                    initial={{ scale: 0, opacity: 0 }}
                    animate={inView ? { 
                      scale: isSelected ? 1.2 : 1, 
                      opacity: 1,
                      // Subtle 'breath' animation to signal interactivity
                      ...( !selected ? {
                        scale: [1, 1.1, 1],
                      } : {})
                    } : { scale: 0, opacity: 0 }}
                    transition={{ 
                      type: 'spring', 
                      stiffness: 300, 
                      damping: 25,
                      delay: isSelected ? 0 : (inView ? (nd.depth * 0.15 + nd.id * 0.006) : 0),
                      // Transition for the 'breath' animation
                      scale: !selected ? {
                        duration: 3,
                        repeat: Infinity,
                        ease: "easeInOut",
                        delay: nd.id * 0.1
                      } : undefined
                    }}
                    whileHover={{ 
                      scale: 1.4, 
                      fillOpacity: 1,
                    }}
                    whileTap={{ scale: 0.9 }}
                    style={{ cursor: 'pointer', transformOrigin: `${node.x!}px ${node.y!}px` }}
                    onClick={() => setSelected(isSelected ? null : { id: nd.id, x: node.x!, y: node.y! })}
                  />
                  {/* Token label for depth-1 roots */}
                  {nd.depth === 1 && inView && (
                    <motion.text
                      x={node.x!}
                      y={(node.y ?? 0) - 14}
                      textAnchor="middle"
                      fill={color}
                      fillOpacity={dimmed ? 0.2 : 1}
                      fontSize={9}
                      fontFamily="JetBrains Mono, monospace"
                      fontWeight="600"
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: -16 }}
                      transition={{ delay: 0.6 + nd.id * 0.05 }}
                    >
                      {token}
                    </motion.text>
                  )}
                </g>
              )
            })}

            {/* Selected node popup */}
            {selected && (() => {
              const nd = TREE_NODES[selected.id]
              const token = TREE_NODE_TOKENS[selected.id]
              const tipX = Math.min(selected.x + 14, SVG_W - 150)
              const tipY = Math.max(selected.y - 50, 10)
              return (
                <g>
                  <rect x={tipX} y={tipY} width={140} height={54} rx={4}
                    fill="white" stroke="#DDD8CE" strokeWidth={1}
                    filter="drop-shadow(0 2px 6px rgba(0,0,0,0.08))" />
                  <text x={tipX + 10} y={tipY + 16} fill="#1A1714" fontSize={11} fontFamily="JetBrains Mono, monospace" fontWeight="600">
                    {token ? `"${token}"` : `node ${selected.id}`}
                  </text>
                  <text x={tipX + 10} y={tipY + 30} fill="#6B6460" fontSize={9} fontFamily="JetBrains Mono, monospace">
                    path: [{nd.path.join(', ')}]
                  </text>
                  <text x={tipX + 10} y={tipY + 44} fill={getDepthColor(nd.depth)} fontSize={9} fontFamily="JetBrains Mono, monospace">
                    depth {nd.depth} · head {nd.depth - 1} · idx {selected.id}
                  </text>
                </g>
              )
            })()}
          </g>
        </svg>
      </div>
      <p className="text-xs font-mono text-ink-soft mt-2 text-center">
        20 candidate tokens — click any node to inspect its path
      </p>
    </div>
  )
}
