import { motion } from 'framer-motion'
import { useScrollSection } from '../../hooks/useScrollSection'
import MedusaArchDiagram from '../viz/MedusaArchDiagram'
import { fadeUp, staggerContainer } from '../../lib/variants'
import { getDepthColor } from '../../data/tree'

export default function ArchitectureSection() {
  const { ref, inView } = useScrollSection(0.2)

  return (
    <section id="section-1" className="min-h-screen flex flex-col justify-center px-8 py-20 bg-white border-t border-birch">
      <div className="max-w-5xl mx-auto w-full">
        <motion.div
          ref={ref as React.Ref<HTMLDivElement>}
          variants={staggerContainer}
          initial="hidden"
          animate={inView ? 'visible' : 'hidden'}
          className="flex flex-col gap-10"
        >
          <motion.div variants={fadeUp} className="text-center">
            <div className="font-mono text-xs text-grove border-b border-grove pb-0.5 mb-4 inline-block tracking-widest uppercase">
              §01 — Architecture
            </div>
            <h2 className="text-3xl font-bold text-ink">
              Four extra heads, frozen backbone
            </h2>
            <p className="mt-3 text-ink-soft max-w-2xl mx-auto">
              MEDUSA attaches K=4 lightweight decoding heads to the final layer of a frozen backbone LLM.
              Only the heads are trained — the backbone never changes.
            </p>
          </motion.div>

          <div className="flex justify-center">
            <motion.div variants={fadeUp} className="max-w-2xl w-full">
              <MedusaArchDiagram inView={inView} />
            </motion.div>
          </div>

          <motion.div variants={fadeUp} className="grid grid-cols-2 gap-4 text-center text-sm max-w-3xl mx-auto w-full">
            {[
              { label: 'Trainable params', value: '~4M', desc: 'Only the 4 heads (vs 7B backbone)' },
              { label: 'Training data', value: '60k', desc: 'ShareGPT assistant turns' },
            ].map(({ label, value, desc }) => (
              <div key={label} className="bg-white border border-birch rounded p-4">
                <div className="text-2xl font-bold text-grove font-mono">{value}</div>
                <div className="font-medium text-ink mt-1 text-sm">{label}</div>
                <div className="text-xs text-ink-soft mt-0.5">{desc}</div>
              </div>
            ))}
          </motion.div>
        </motion.div>
      </div>
    </section>
  )
}
