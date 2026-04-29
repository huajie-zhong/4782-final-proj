import { useState } from 'react'
import { motion } from 'framer-motion'
import { useScrollSection } from '../../hooks/useScrollSection'
import HeadProposalViz from '../viz/HeadProposalViz'
import { fadeUp, staggerContainer } from '../../lib/variants'
import { EXAMPLE_PROMPT } from '../../data/tokens'
import { getDepthColor } from '../../data/tree'

const HEAD_LABELS = ['Head 0', 'Head 1', 'Head 2', 'Head 3']
const HEAD_OFFSETS = ['t+2', 't+3', 't+4', 't+5']
const S_K = [10, 3, 2, 2]

export default function ProposalSection() {
  const { ref, inView } = useScrollSection(0.2)
  const [selectedHead, setSelectedHead] = useState<number | null>(null)

  return (
    <section id="section-2" className="min-h-screen flex flex-col justify-center border-t border-birch">
      <div className="max-w-5xl mx-auto w-full">
        <motion.div
          ref={ref as React.Ref<HTMLDivElement>}
          variants={staggerContainer}
          initial="hidden"
          animate={inView ? 'visible' : 'hidden'}
          className="flex flex-col gap-8"
        >
          <motion.div variants={fadeUp} className="text-center">
            <div className="font-mono text-[10px] md:text-xs text-grove border-b border-grove pb-0.5 mb-4 inline-block tracking-widest uppercase">
              §02 — Token Proposals
            </div>
            <h2 className="text-2xl md:text-3xl font-bold text-ink">
              Each head proposes tokens at a different depth
            </h2>
            <p className="mt-3 text-xs md:text-base text-ink-soft max-w-2xl mx-auto">
              After one forward pass, the backbone returns hidden states. Each Medusa head applies
              its learned transformation and produces a probability distribution over the vocabulary —
              targeting a different future position.
            </p>
          </motion.div>
 
          {/* Prompt context */}
          <motion.div variants={fadeUp} className="bg-white border border-birch rounded p-4">
            <div className="text-[10px] font-mono text-ink-soft mb-2 uppercase tracking-tight">Current prompt (example):</div>
            <div className="font-mono text-xs md:text-sm text-ink overflow-x-auto whitespace-nowrap md:whitespace-normal pb-2 md:pb-0">
              <span className="px-2 py-1 bg-parchment rounded border border-birch">{EXAMPLE_PROMPT}</span>
            </div>
            <div className="text-[10px] md:text-xs text-ink-soft mt-2">
              The backbone processes this and returns hidden states h at the last token position.
              All 4 heads then run simultaneously on h.
            </div>
          </motion.div>
 
          {/* Head selector buttons */}
          <motion.div variants={fadeUp} className="flex flex-col md:flex-row md:items-center gap-3">
            <span className="text-[10px] md:text-xs font-mono text-ink-soft md:mr-1">Click to focus a head:</span>
            <div className="flex flex-wrap gap-2">
              {HEAD_LABELS.map((label, k) => {
                const color = getDepthColor(k + 1)
                const active = selectedHead === k
                return (
                  <button
                    key={k}
                    onClick={() => setSelectedHead(active ? null : k)}
                    className="flex-1 md:flex-none px-2 md:px-3 py-1.5 text-[9px] md:text-xs font-mono rounded border transition-all duration-150 whitespace-nowrap"
                    style={{
                      borderColor: color,
                      backgroundColor: active ? color : 'transparent',
                      color: active ? 'white' : color,
                    }}
                  >
                    {label} · t+{k+2}
                  </button>
                )
              })}
            </div>
          </motion.div>

          <motion.div variants={fadeUp}>
            <HeadProposalViz inView={inView} selectedHead={selectedHead} />
          </motion.div>

          <motion.div variants={fadeUp} className="border-l-2 border-grove pl-4 text-sm text-ink-soft">
            The top-k tokens from each head get combined into a <strong className="text-ink">candidate tree</strong>.
            The top proposal from Head 0 forms the root, while heads 1-3 form the subsequent branches —
            up to 20 candidate paths verified in one pass.{' '}
            <a href="#section-3" className="text-grove underline underline-offset-2 hover:text-ink transition-colors" onClick={e => { e.preventDefault(); document.getElementById('section-3')?.scrollIntoView({ behavior: 'smooth' }) }}>
              See the tree →
            </a>
          </motion.div>
        </motion.div>
      </div>
    </section>
  )
}
