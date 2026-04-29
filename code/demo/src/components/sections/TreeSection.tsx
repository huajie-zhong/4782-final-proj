import { motion } from 'framer-motion'
import { useScrollSection } from '../../hooks/useScrollSection'
import CandidateTreeViz from '../viz/CandidateTreeViz'
import { fadeUp, staggerContainer } from '../../lib/variants'
import { DEPTH_COUNTS, S_K, getDepthColor } from '../../data/tree'

export default function TreeSection() {
  const { ref, inView } = useScrollSection(0.15)

  return (
    <section id="section-3" className="min-h-screen flex flex-col justify-center bg-white border-t border-birch">
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
              §03 — Candidate Tree
            </div>
            <h2 className="text-2xl md:text-3xl font-bold text-ink">
              20 candidate paths, one tree
            </h2>
            <p className="mt-3 text-sm md:text-base text-ink-soft max-w-2xl mx-auto">
              The top-k tokens from each head are assembled into a static BFS-ordered tree.
              Each path from root to leaf represents a candidate sequence of 1–4 tokens.
              Click any node to inspect its path.
            </p>
          </motion.div>
 
          {/* Depth stats */}
          <motion.div variants={fadeUp} className="grid grid-cols-2 md:grid-cols-4 gap-3 text-center">
            {[1, 2, 3, 4].map(d => (
              <div key={d} className="bg-parchment border border-birch rounded p-3">
                <div className="text-lg md:text-xl font-bold font-mono" style={{ color: getDepthColor(d) }}>
                  {DEPTH_COUNTS[d - 1]}
                </div>
                <div className="text-[10px] md:text-xs text-ink-soft mt-0.5">depth {d} nodes</div>
                <div className="text-[9px] md:text-xs text-birch font-mono">top-{S_K[d - 1]}/parent</div>
              </div>
            ))}
          </motion.div>

          <motion.div variants={fadeUp}>
            <CandidateTreeViz inView={inView} />
          </motion.div>

          <motion.div variants={fadeUp} className="border-l-2 border-birch pl-4 text-xs text-ink-soft leading-relaxed">
            Nodes are stored breadth-first: the depth-1 root first (index 0), then depth-2 (1–3),
            depth-3 (4–9), depth-4 (10–19). BFS ordering guarantees every parent index is less than
            its children's — so any budget ≤ 20 can be applied by simple truncation without breaking ancestry.
          </motion.div>
        </motion.div>
      </div>
    </section>
  )
}
