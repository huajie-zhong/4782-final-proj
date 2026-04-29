import { motion, AnimatePresence } from 'framer-motion'
import { HEAD_PROPOSALS, HEAD_PROPOSALS_1, HEAD_PROPOSALS_2, HEAD_PROPOSALS_3 } from '../../data/tokens'
import { getDepthColor } from '../../data/tree'
import { staggerContainer, fadeUp } from '../../lib/variants'

const HEADS = [
  { k: 0, label: 'Head 0', offset: 't+2', proposals: HEAD_PROPOSALS as unknown as {token:string,prob:number,rank:number}[] },
  { k: 1, label: 'Head 1', offset: 't+3', proposals: HEAD_PROPOSALS_1 as unknown as {token:string,prob:number,rank:number}[] },
  { k: 2, label: 'Head 2', offset: 't+4', proposals: HEAD_PROPOSALS_2 as unknown as {token:string,prob:number,rank:number}[] },
  { k: 3, label: 'Head 3', offset: 't+5', proposals: HEAD_PROPOSALS_3 as unknown as {token:string,prob:number,rank:number}[] },
]

type Props = { inView: boolean; selectedHead: number | null }

export default function HeadProposalViz({ inView, selectedHead }: Props) {
  const visibleHeads = selectedHead !== null ? HEADS.filter(h => h.k === selectedHead) : HEADS

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={selectedHead ?? 'all'}
        variants={staggerContainer}
        initial="hidden"
        animate={inView ? 'visible' : 'hidden'}
        exit={{ opacity: 0 }}
        className={`grid gap-4 w-full ${selectedHead !== null ? 'grid-cols-1 max-w-sm mx-auto' : 'grid-cols-1 sm:grid-cols-2 md:grid-cols-4'}`}
      >
        {visibleHeads.map(({ k, label, offset, proposals }) => {
          const color = getDepthColor(k + 1)
          return (
            <motion.div key={k} variants={fadeUp} className="flex flex-col gap-2">
              {/* Header */}
              <div
                className="rounded p-3 text-center border"
                style={{ borderColor: `${color}60`, backgroundColor: `${color}0d` }}
              >
                <div className="font-bold font-mono text-sm" style={{ color }}>{label}</div>
                <div className="text-xs text-ink-soft font-mono mt-0.5">position {offset}</div>
                <div className="text-[11px] text-ink-soft mt-1">top-{proposals.length} candidates</div>
              </div>
              {/* Token bars */}
              <div className="flex flex-col gap-1.5">
                {proposals.map(({ token, prob, rank }) => (
                  <div key={rank} className="flex flex-col gap-0.5">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-mono text-ink px-1.5 py-0.5 bg-parchment border border-birch rounded">
                        {token}
                      </span>
                      <span className="text-ink-soft font-mono text-[10px]">{(prob * 100).toFixed(0)}%</span>
                    </div>
                    <div className="h-1.5 bg-parchment rounded-sm overflow-hidden">
                      <motion.div
                        initial={{ scaleX: 0 }}
                        animate={inView ? { scaleX: 1 } : { scaleX: 0 }}
                        transition={{ duration: 0.5, delay: k * 0.1 + rank * 0.05, ease: 'easeOut' }}
                        style={{ backgroundColor: color, originX: 0, width: `${prob * 100}%` }}
                        className="h-full rounded-sm"
                      />
                    </div>
                  </div>
                ))}
              </div>
              {/* Rank badge */}
              <div
                className="text-[10px] text-center py-1 rounded border font-mono"
                style={{ borderColor: `${color}50`, color, backgroundColor: `${color}0a` }}
              >
                rank 0 → tree depth {k + 1}
              </div>
            </motion.div>
          )
        })}
      </motion.div>
    </AnimatePresence>
  )
}
