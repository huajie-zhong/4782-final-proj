import { useRef, useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import * as d3 from 'd3'
import { TREE_NODES, getDepthColor } from '../../data/tree'
import { TREE_NODE_TOKENS } from '../../data/tokens'
import { useTreeLayout } from '../../hooks/useTreeLayout'

const SVG_W = 900
const SVG_H = 320
const GPU_X = SVG_W / 2
const GPU_Y = -40

type Props = { inView: boolean }

export default function VerifyPassViz({ inView }: Props) {
  const [pulse, setPulse] = useState(0)
  const [nodesLit, setNodesLit] = useState<Set<number>>(new Set())
  const root = useTreeLayout(TREE_NODES, SVG_W - 60, SVG_H - 60)

  useEffect(() => {
    if (!inView) return
    let pulseCount = 0
    const interval = setInterval(() => {
      pulseCount++
      setPulse(p => p + 1)
      // Light up all nodes
      setTimeout(() => {
        setNodesLit(new Set(TREE_NODES.map(n => n.id)))
        setTimeout(() => setNodesLit(new Set()), 600)
      }, 400)
    }, 2200)
    return () => clearInterval(interval)
  }, [inView])

  const nodeDescendants = root.descendants().filter(d => d.data.data !== null)
  const links = root.links().filter(l => l.source.data.data !== null)

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div className="bg-white border border-birch rounded p-4 text-center">
          <div className="text-2xl font-bold font-mono text-ink">64</div>
          <div className="text-ink font-medium text-sm">forward passes needed</div>
          <div className="text-xs text-ink-soft mt-1">Autoregressive decoding</div>
        </div>
        <div className="bg-grove-light border border-grove rounded p-4 text-center">
          <div className="text-2xl font-bold font-mono text-grove">1</div>
          <div className="text-grove font-medium text-sm">forward pass needed</div>
          <div className="text-xs text-grove mt-1">MEDUSA tree verification</div>
        </div>
      </div>

      <div className="overflow-x-auto border border-birch rounded bg-parchment relative">
        {/* GPU icon at top */}
        <div className="absolute top-2 left-1/2 -translate-x-1/2 z-10">
          <div className="px-3 py-1.5 bg-ink text-parchment text-xs rounded font-mono">
            GPU · forward pass
          </div>
        </div>

        <svg width={SVG_W} height={SVG_H + 20} style={{ display: 'block', minWidth: SVG_W }}>
          <defs>
            <radialGradient id="pulseGrad" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#6366f1" stopOpacity="0.6" />
              <stop offset="100%" stopColor="#6366f1" stopOpacity="0" />
            </radialGradient>
          </defs>
          <g transform="translate(30, 60)">
            {/* Pulse waves from GPU */}
            <AnimatePresence>
              {inView && (
                <motion.circle
                  key={pulse}
                  cx={GPU_X - 30}
                  cy={GPU_Y + 60}
                  r={0}
                  fill="url(#pulseGrad)"
                  initial={{ r: 0, opacity: 0.7 }}
                  animate={{ r: SVG_W * 0.8, opacity: 0 }}
                  transition={{ duration: 1.5, ease: 'easeOut' }}
                />
              )}
            </AnimatePresence>

            {/* Links */}
            {links.map((link, i) => {
              const src = link.source as d3.HierarchyPointNode<typeof link.source.data>
              const tgt = link.target as d3.HierarchyPointNode<typeof link.target.data>
              const depth = tgt.data.data?.depth ?? 1
              return (
                <line
                  key={i}
                  x1={src.x ?? 0} y1={src.y ?? 0}
                  x2={tgt.x ?? 0} y2={tgt.y ?? 0}
                  stroke={getDepthColor(depth)}
                  strokeOpacity={0.3}
                  strokeWidth={1}
                />
              )
            })}

            {/* Nodes */}
            {nodeDescendants.map(node => {
              const nd = node.data.data!
              const isLit = nodesLit.has(nd.id)
              const color = getDepthColor(nd.depth)
              return (
                <g key={nd.id}>
                  {isLit && (
                    <circle
                      cx={node.x ?? 0}
                      cy={node.y ?? 0}
                      r={14}
                      fill={color}
                      opacity={0.25}
                    />
                  )}
                  <motion.circle
                    cx={node.x ?? 0}
                    cy={node.y ?? 0}
                    r={nd.depth === 1 ? 7 : 4.5}
                    fill={isLit ? color : color}
                    fillOpacity={isLit ? 1 : 0.6}
                    stroke={isLit ? 'white' : 'white'}
                    strokeWidth={1.5}
                    animate={{
                      scale: isLit ? [1, 1.4, 1] : 1,
                      fillOpacity: isLit ? [0.6, 1, 0.6] : 0.6,
                    }}
                    transition={{ duration: 0.3 }}
                    style={{ transformOrigin: `${node.x ?? 0}px ${node.y ?? 0}px` }}
                  />
                  {nd.depth === 1 && (
                    <text
                      x={node.x ?? 0}
                      y={(node.y ?? 0) - 12}
                      textAnchor="middle"
                      fill={color}
                      fontSize={8}
                      fontFamily="JetBrains Mono, monospace"
                    >
                      {TREE_NODE_TOKENS[nd.id]}
                    </text>
                  )}
                </g>
              )
            })}

            {/* Lines from GPU to root nodes */}
            {nodeDescendants
              .filter(n => n.data.data?.depth === 1)
              .map(node => (
                <motion.line
                  key={`gpu-${node.data.data?.id}`}
                  x1={GPU_X - 30}
                  y1={GPU_Y + 60}
                  x2={node.x ?? 0}
                  y2={node.y ?? 0}
                  stroke="#6366f1"
                  strokeWidth={0.8}
                  strokeDasharray="3,4"
                  strokeOpacity={0.4}
                  animate={{ strokeOpacity: [0.2, 0.6, 0.2] }}
                  transition={{ repeat: Infinity, duration: 2, delay: (node.data.data?.id ?? 0) * 0.1 }}
                />
              ))}
          </g>
        </svg>
      </div>
      <p className="text-xs text-center font-mono text-ink-soft">
        One forward pass with tree-attention mask simultaneously computes logits for all 64 candidates
      </p>
    </div>
  )
}
