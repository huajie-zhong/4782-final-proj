import { motion } from 'framer-motion'
import { useScrollSection } from '../../hooks/useScrollSection'
import AutoregressiveViz from '../viz/AutoregressiveViz'
import { fadeUp, staggerContainer } from '../../lib/variants'

export default function HeroSection() {
  const { ref, inView } = useScrollSection(0.3)

  return (
    <section id="section-0" className="min-h-screen flex flex-col justify-center max-w-5xl mx-auto">
      <motion.div
        ref={ref as React.Ref<HTMLDivElement>}
        variants={staggerContainer}
        initial="hidden"
        animate={inView ? 'visible' : 'hidden'}
        className="flex flex-col gap-8"
      >
        <motion.div variants={fadeUp} className="text-center">
          <div className="font-mono text-[10px] md:text-xs text-grove border-b border-grove pb-0.5 mb-5 inline-block tracking-widest uppercase">
            §00 — CS 4782 · MEDUSA Speculative Decoding
          </div>
          <h1 className="text-3xl md:text-5xl font-extrabold text-ink leading-tight">
            What if your LLM could<br />
            <span className="text-grove">predict 4 tokens at once?</span>
          </h1>
          <p className="mt-4 text-base md:text-lg text-ink-soft max-w-2xl mx-auto">
            Large language models generate text one token at a time — an inherently sequential bottleneck.
            MEDUSA breaks this limit by adding speculative decoding heads that propose multiple future tokens
            simultaneously, then verifying them in a single forward pass.
          </p>
        </motion.div>
 
        <motion.div variants={fadeUp} className="grid grid-cols-2 md:flex md:justify-center gap-6 md:gap-10">
          {[
            { value: '2.2×', label: 'speedup on Vicuna-7B', color: 'text-grove' },
            { value: '4', label: 'decoding heads', color: 'text-depth-1' },
            { value: '64', label: 'candidates per pass', color: 'text-depth-3' },
            { value: '0', label: 'backbone params changed', color: 'text-ink-soft' },
          ].map(({ value, label, color }) => (
            <div key={label} className="text-center">
              <div className={`text-2xl md:text-4xl font-bold font-mono ${color}`}>{value}</div>
              <div className="text-[10px] md:text-xs text-ink-soft mt-1">{label}</div>
            </div>
          ))}
        </motion.div>

        <motion.div variants={fadeUp}>
          <div className="text-xs font-mono text-ink-soft uppercase tracking-wider mb-3 text-center">
            live comparison — same prompt, same model
          </div>
          <AutoregressiveViz active={inView} />
        </motion.div>

        <motion.div variants={fadeUp} className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <a
            href="#live-demo"
            onClick={(e) => {
              e.preventDefault()
              document.getElementById('live-demo')?.scrollIntoView({ behavior: 'smooth' })
            }}
            className="px-6 py-2.5 rounded bg-grove text-white text-sm font-mono tracking-wide hover:opacity-80 transition-opacity"
          >
            Try it yourself →
          </a>
          <a
            href="/how-it-works"
            className="text-sm font-mono text-ink-soft hover:text-ink transition-colors"
          >
            How it works ↓
          </a>
        </motion.div>
      </motion.div>
    </section>
  )
}
