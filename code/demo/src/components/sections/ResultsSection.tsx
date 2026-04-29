import { motion } from 'framer-motion'
import { useScrollSection } from '../../hooks/useScrollSection'
import SpeedupCharts from '../viz/SpeedupCharts'
import { fadeUp, staggerContainer } from '../../lib/variants'
import { BENCHMARK } from '../../data/benchmarks'

export default function ResultsSection() {
  const { ref, inView } = useScrollSection(0.1)

  return (
    <section id="section-7" className="min-h-screen flex flex-col justify-center px-8 py-20 bg-parchment border-t border-birch">
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
              §07 — Real Results
            </div>
            <h2 className="text-3xl font-bold text-ink">
              2.2× on real hardware
            </h2>
            <p className="mt-3 text-ink-soft max-w-2xl mx-auto">
              Measured on TinyLlama-1.1B-Chat. All numbers from real benchmark runs, not estimates.
              Paper target (Vicuna-7B) achieves 2.18× at the same relative speedup.
            </p>
          </motion.div>

          {/* Big numbers */}
          <motion.div variants={fadeUp} className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            {[
              { value: `${BENCHMARK.greedyTps.toFixed(1)}`, unit: 'tok/s', label: 'Baseline' },
              { value: `${BENCHMARK.medusaTypicalTps.toFixed(1)}`, unit: 'tok/s', label: 'MEDUSA + typical' },
              { value: `${BENCHMARK.speedupRatioTypical.toFixed(2)}×`, unit: '', label: 'Wall-clock speedup' },
              { value: `${(BENCHMARK.avgAcceptanceRateTypical).toFixed(2)}`, unit: 'tok/pass', label: 'Avg tokens accepted' },
            ].map(({ value, unit, label }) => (
              <div key={label} className="bg-white border border-birch rounded p-4">
                <div className="text-3xl font-bold font-mono text-grove">{value}<span className="text-lg ml-0.5 text-ink-soft">{unit}</span></div>
                <div className="text-xs text-ink-soft mt-1">{label}</div>
              </div>
            ))}
          </motion.div>

          <motion.div variants={fadeUp}>
            <SpeedupCharts inView={inView} />
          </motion.div>

          <motion.div variants={fadeUp} className="text-center pt-4 border-t border-birch">
            <p className="text-ink-soft text-sm">
              CS 4782 Final Project · MEDUSA Speculative Decoding Reproduction
            </p>
            <p className="text-birch text-xs mt-1">
              Paper: <em>MEDUSA: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads</em> (Cai et al., 2024)
            </p>
          </motion.div>
        </motion.div>
      </div>
    </section>
  )
}
