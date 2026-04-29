import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import { motion, AnimatePresence } from 'framer-motion'
import { MEDUSA_TREE_PARENT_INDICES, computeAttentionMatrix, TREE_NODES, getDepthColor } from '../../data/tree'
import { GREEDY_ACCEPTED_PATH } from '../../data/tokens'

const PREFIX_LEN = 8
const TREE_SIZE = 20
const TOTAL_COLS = PREFIX_LEN + TREE_SIZE

const SVG_W = 600
const SVG_H = 400
const CELL_W = SVG_W / TOTAL_COLS
const CELL_H = SVG_H / TREE_SIZE

const MATRIX = computeAttentionMatrix(MEDUSA_TREE_PARENT_INDICES as number[], PREFIX_LEN)

const MASKED = '#F7F3EC'  // parchment
const TEAL   = '#2dd4bf'
const INDIGO = '#6366f1'
const LIGHT_INDIGO = '#a5b4fc'

function computeCellFill(r: number, c: number, step: number): string {
  if (!MATRIX[r][c]) return MASKED
  if (c < PREFIX_LEN) {
    return step >= 1 ? TEAL : MASKED
  }
  const tc = c - PREFIX_LEN
  if (tc === r) return step >= 2 ? INDIGO : MASKED
  const depth = TREE_NODES[r]?.depth ?? 1
  if (depth >= 2 && step >= 4) return LIGHT_INDIGO
  if (depth >= 3 && step >= 5) return LIGHT_INDIGO
  return MASKED
}

function isCellNewlyActive(r: number, c: number, step: number, prevStep: number): boolean {
  return computeCellFill(r, c, step) !== MASKED && computeCellFill(r, c, prevStep) === MASKED
}

type Props = { step: number }

export default function AttentionMaskViz({ step }: Props) {
  const matrixRef = useRef<SVGGElement>(null)
  const prevStepRef = useRef(-1)
  const [hovered, setHovered] = useState<{ row: number; col: number } | null>(null)
  const [pulseKey, setPulseKey] = useState(0)

  useEffect(() => {
    if (step === 7) {
      setPulseKey(prev => prev + 1)
    }
  }, [step])

  useEffect(() => {
    if (!matrixRef.current) return
    const g = d3.select(matrixRef.current)
    g.selectAll('*').remove()
 
    const prevStep = prevStepRef.current
    prevStepRef.current = step
 
    // Matrix cells

    // Build flat cell data
    const flatData: { r: number; c: number; val: boolean }[] = []
    for (let r = 0; r < TREE_SIZE; r++) {
      for (let c = 0; c < TOTAL_COLS; c++) {
        flatData.push({ r, c, val: MATRIX[r][c] })
      }
    }

    // Draw cells
    const cells = g.selectAll<SVGRectElement, { r: number; c: number; val: boolean }>('rect.cell')
      .data(flatData)
      .enter()
      .append('rect')
      .attr('class', 'cell')
      .attr('x', d => d.c * CELL_W)
      .attr('y', d => d.r * CELL_H)
      .attr('width', CELL_W - 0.3)
      .attr('height', CELL_H - 0.3)
      .attr('rx', 0.4)
      .attr('fill', d => computeCellFill(d.r, d.c, Math.max(0, prevStep)))
      .on('mouseover', function(_, d) {
        if (step < 6) return
        setHovered({ row: d.r, col: d.c })
        g.selectAll<SVGRectElement, { r: number; c: number; val: boolean }>('rect.cell')
          .attr('opacity', cell => {
            if (cell.r === d.r) return 1
            if (cell.c === d.c && d.c >= PREFIX_LEN) return 1
            return 0.2
          })
        d3.select<SVGRectElement, { r: number; c: number; val: boolean }>(this)
          .attr('stroke', '#f59e0b')
          .attr('stroke-width', 1.5)
      })
      .on('mouseout', function() {
        g.selectAll('rect.cell').attr('opacity', 1).attr('stroke', 'none')
        setHovered(null)
      })

    // Transitions
    const increasing = step > prevStep
    cells
      .transition()
      .duration(increasing ? 180 : 80)
      .delay(d => {
        if (!increasing) return 0
        if (!isCellNewlyActive(d.r, d.c, step, prevStep)) return 0
        if (step === 1) return d.r * 5
        if (step === 2) return d.r * 8
        if (step >= 4) return d.r * 5
        return 0
      })
      .attr('fill', d => computeCellFill(d.r, d.c, step))

    // Winning path highlight
    if (step >= 9) {
      GREEDY_ACCEPTED_PATH.forEach(rowIdx => {
        g.append('rect')
          .attr('x', -2)
          .attr('y', rowIdx * CELL_H - 1)
          .attr('width', TOTAL_COLS * CELL_W + 4)
          .attr('height', CELL_H + 1)
          .attr('fill', 'none')
          .attr('stroke', '#10b981')
          .attr('stroke-width', 1.5)
          .attr('opacity', 0.6)
          .attr('rx', 2)
      })
    }

    // Output column
    if (step >= 8) {
      const outputX = TOTAL_COLS * CELL_W + 25
      const outG = g.append('g').attr('transform', `translate(${outputX}, 0)`)
      
      outG.append('text')
        .attr('y', -8)
        .attr('fill', '#94a3b8')
        .attr('font-size', 8)
        .attr('font-family', 'JetBrains Mono, monospace')
        .text('Logit Matches')

      for (let r = 0; r < TREE_SIZE; r++) {
        const isAccepted = GREEDY_ACCEPTED_PATH.includes(r)
        const isMismatch = !isAccepted && step >= 9
        
        outG.append('rect')
          .attr('y', r * CELL_H)
          .attr('width', 24)
          .attr('height', CELL_H - 1)
          .attr('fill', isAccepted ? '#10b981' : (isMismatch ? '#ef4444' : '#E2E8F0'))
          .attr('rx', 1.5)
          .attr('opacity', step >= 9 ? 0.8 : 0.3)
          
        if (step >= 9) {
          outG.append('text')
            .attr('x', 28)
            .attr('y', r * CELL_H + CELL_H / 2 + 3)
            .attr('fill', isAccepted ? '#059669' : '#ef4444')
            .attr('font-size', 8)
            .attr('font-family', 'Inter')
            .attr('font-weight', 'bold')
            .text(isAccepted ? '✓' : '✗')
        }
      }
    }

    // Depth color indicators
    for (let r = 0; r < TREE_SIZE; r++) {
      const depth = TREE_NODES[r]?.depth ?? 1
      g.append('rect')
        .attr('x', -8)
        .attr('y', r * CELL_H)
        .attr('width', 4)
        .attr('height', CELL_H - 0.3)
        .attr('fill', getDepthColor(depth))
        .attr('opacity', 0.6)
    }

    // Annotations
    if (step >= 1) {
      const separatorX = PREFIX_LEN * CELL_W
      g.append('line')
        .attr('x1', separatorX)
        .attr('y1', -16)
        .attr('x2', separatorX)
        .attr('y2', TREE_SIZE * CELL_H)
        .attr('stroke', '#DDD8CE')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '3,2')
    }
  }, [step])

  return (
    <div className="flex flex-col gap-3">
      <div className="overflow-x-auto rounded border border-birch bg-white p-4 pb-2 relative">
        <div className="flex gap-3 text-xs mb-3 font-mono">
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-3 rounded-sm" style={{ backgroundColor: TEAL }} />
            prefix
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-3 rounded-sm" style={{ backgroundColor: INDIGO }} />
            self
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-3 rounded-sm" style={{ backgroundColor: LIGHT_INDIGO }} />
            ancestor
          </span>
          {step >= 8 && (
            <span className="flex items-center gap-1">
              <span className="inline-block w-3 h-3 rounded-sm" style={{ backgroundColor: '#10b981' }} />
              match
            </span>
          )}
          {hovered && (
            <span className="ml-auto text-ink-soft">
              row {hovered.row} (depth {TREE_NODES[hovered.row]?.depth}) — col {hovered.col}
            </span>
          )}
        </div>

        <div className="relative w-full aspect-[6/4.5] md:aspect-auto">
          <svg
            viewBox={`0 0 ${SVG_W + 100} ${SVG_H + 28}`}
            className="w-full h-auto"
            style={{ display: 'block' }}
          >
            {/* Matrix Layer (Managed by D3) */}
            <g ref={matrixRef} transform="translate(10, 20)" />

            {/* Pulse Overlay (Managed by React, Internal to SVG for alignment) */}
            <AnimatePresence>
              {step === 7 && (
                <motion.rect
                  key={pulseKey}
                  x={10}
                  y={20}
                  width={2}
                  height={SVG_H}
                  fill="#6366f1"
                  style={{ filter: 'drop-shadow(0 0 8px #6366f1)' }}
                  initial={{ x: 10, opacity: 0 }}
                  animate={{ x: [10, 10 + SVG_W], opacity: [0, 1, 1, 0] }}
                  transition={{ duration: 1.5, ease: 'linear' }}
                />
              )}
            </AnimatePresence>
          </svg>
        </div>
      </div>
      <div className="flex justify-between text-xs font-mono text-ink-soft">
        <span>64 candidates × (8 prefix + 64 tree) tokens</span>
        {step >= 9 && (
          <span className="text-grove font-bold"> Longest valid path: {GREEDY_ACCEPTED_PATH.length} tokens verified</span>
        )}
      </div>
    </div>
  )
}
