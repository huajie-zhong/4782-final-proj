import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import { MEDUSA_TREE_PARENT_INDICES, computeAttentionMatrix, TREE_NODES, getDepthColor } from '../../data/tree'

const PREFIX_LEN = 8
const TREE_SIZE = 20
const TOTAL_COLS = PREFIX_LEN + TREE_SIZE

const SVG_W = 600
const SVG_H = 400
const CELL_W = SVG_W / TOTAL_COLS
const CELL_H = SVG_H / TREE_SIZE

const MATRIX = computeAttentionMatrix(MEDUSA_TREE_PARENT_INDICES as number[], PREFIX_LEN)

const MASKED = '#F7F3EC'  // parchment — matches body bg
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
  const svgRef = useRef<SVGSVGElement>(null)
  const prevStepRef = useRef(-1)
  const [hovered, setHovered] = useState<{ row: number; col: number } | null>(null)

  useEffect(() => {
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const prevStep = prevStepRef.current
    prevStepRef.current = step

    const g = svg.append('g').attr('transform', 'translate(10, 20)')

    // Build flat cell data
    const flatData: { r: number; c: number; val: boolean }[] = []
    for (let r = 0; r < TREE_SIZE; r++) {
      for (let c = 0; c < TOTAL_COLS; c++) {
        flatData.push({ r, c, val: MATRIX[r][c] })
      }
    }

    // Draw cells with previous step's fill initially (avoids flash)
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

    // Transition cells to current step's fill with stagger for newly active cells
    const increasing = step > prevStep
    cells
      .transition()
      .duration(increasing ? 180 : 80)
      .delay(d => {
        if (!increasing) return 0
        if (!isCellNewlyActive(d.r, d.c, step, prevStep)) return 0
        if (step === 1) return d.r * 5          // row wave: prefix flood
        if (step === 2) return d.r * 8          // diagonal sweep
        if (step >= 4) return d.r * 5           // ancestor sweep
        return 0
      })
      .attr('fill', d => computeCellFill(d.r, d.c, step))

    // Depth color indicators on left side
    for (let r = 0; r < TREE_SIZE; r++) {
      const depth = TREE_NODES[r]?.depth ?? 1
      g.append('rect')
        .attr('x', -6)
        .attr('y', r * CELL_H)
        .attr('width', 4)
        .attr('height', CELL_H - 0.3)
        .attr('fill', getDepthColor(depth))
        .attr('opacity', 0.6)
    }

    // Column annotations on top
    if (step >= 1) {
      for (let c = 0; c < TOTAL_COLS; c++) {
        if (c < PREFIX_LEN) {
          if (c === 0) {
            g.append('text')
              .attr('x', c * CELL_W + CELL_W / 2)
              .attr('y', -8)
              .attr('text-anchor', 'middle')
              .attr('fill', '#0d9488')
              .attr('font-size', 8)
              .attr('font-family', 'JetBrains Mono, monospace')
              .text('prefix')
          }
        } else {
          const tc = c - PREFIX_LEN
          const depth = TREE_NODES[tc]?.depth ?? 1
          if (tc % 8 === 0) {
            g.append('text')
              .attr('x', c * CELL_W + CELL_W / 2)
              .attr('y', -8)
              .attr('text-anchor', 'middle')
              .attr('fill', getDepthColor(depth))
              .attr('font-size', 7)
              .attr('font-family', 'JetBrains Mono, monospace')
              .text(`P+${depth - 1}`)
          }
        }
      }
    }

    // Prefix/tree separator line
    if (step >= 1) {
      g.append('line')
        .attr('x1', PREFIX_LEN * CELL_W)
        .attr('y1', -16)
        .attr('x2', PREFIX_LEN * CELL_W)
        .attr('y2', TREE_SIZE * CELL_H)
        .attr('stroke', '#DDD8CE')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '3,2')
    }
  }, [step])

  return (
    <div className="flex flex-col gap-3">
      <div className="overflow-x-auto rounded border border-birch bg-white p-4 pb-2">
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
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-3 rounded-sm bg-parchment border border-birch" />
            masked
          </span>
          {hovered && (
            <span className="ml-auto text-ink-soft">
              row {hovered.row} (depth {TREE_NODES[hovered.row]?.depth}) — col {hovered.col}
            </span>
          )}
        </div>
        <svg
          ref={svgRef}
          width={SVG_W + 16}
          height={SVG_H + 28}
          style={{ display: 'block', overflow: 'visible' }}
        />
      </div>
      <div className="text-xs font-mono text-ink-soft text-right">
        64 rows × 72 cols (8 prefix + 64 tree) = 4,608 attention pairs
      </div>
    </div>
  )
}
