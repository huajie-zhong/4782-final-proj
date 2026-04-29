import { useRef, useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import AttentionMaskViz from '../viz/AttentionMaskViz'

const STEP_INFO = [
  {
    title: 'The attention challenge',
    desc: '20 candidate tokens must attend over the full context. But we cannot use standard causal attention — candidates in different branches must not attend to each other.',
  },
  {
    title: 'Attend to the full prefix',
    desc: 'Every candidate node attends to all prefix tokens (the original prompt). All candidates share the same context.',
  },
  {
    title: 'Attend to yourself',
    desc: 'Every node attends to its own position. The diagonal of the right block is always lit.',
  },
  {
    title: 'Root isolation',
    desc: 'The depth-1 root only attends to prefix + self. It starts the independent hypothesis.',
  },
  {
    title: 'Depth-2 nodes see their parent',
    desc: 'A depth-2 node like [0,0] attends to its parent [0]. It knows what token preceded it in this branch.',
  },
  {
    title: 'Full ancestry chains',
    desc: 'Depth-3 and depth-4 nodes trace back through their entire ancestor chain. Each node sees exactly the tokens on its path from root.',
  },
  {
    title: 'Interactive — hover to explore',
    desc: "Hover over any cell to see a node's full attention pattern. Notice: no node attends to siblings or cousins — only ancestors.",
  },
]

export default function AttentionMaskSection() {
  const sectionRef = useRef<HTMLDivElement>(null)
  const [step, setStep] = useState(0)
  const [showInsights, setShowInsights] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      const section = sectionRef.current
      if (!section) return
      const rect = section.getBoundingClientRect()
      const sectionHeight = section.offsetHeight
      const viewH = window.innerHeight
      const progress = Math.max(0, Math.min(1, (-rect.top) / (sectionHeight - viewH)))
      const newStep = Math.min(6, Math.floor(progress * 7))
      setStep(newStep)
    }
    window.addEventListener('scroll', handleScroll, { passive: true })
    handleScroll()
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const info = STEP_INFO[step]

  return (
    <section
      id="section-4"
      ref={sectionRef}
      style={{ height: '600vh' }}
      className="relative border-t border-birch"
    >
      <div className="sticky top-0 h-screen flex flex-col justify-center px-8 py-8 bg-parchment">
        <div className="max-w-5xl mx-auto w-full flex flex-col gap-5">
          {/* Header */}
          <div className="flex items-start justify-between gap-4">
            <div>
              <div className="font-mono text-xs text-grove border-b border-grove pb-0.5 mb-3 inline-block tracking-widest uppercase">
                §04 — Tree Attention Mask
              </div>
              <h2 className="text-2xl font-bold text-ink">{info.title}</h2>
              <p className="mt-1 text-ink-soft text-sm max-w-xl">{info.desc}</p>
            </div>
            {/* Step counter + next button */}
            <div className="flex flex-col items-end gap-2 shrink-0 pt-1">
              <div className="flex gap-1.5">
                {STEP_INFO.map((_, i) => (
                  <div
                    key={i}
                    className="w-1.5 h-1.5 rounded-sm transition-colors duration-200"
                    style={{ backgroundColor: i <= step ? '#2A5C45' : '#DDD8CE' }}
                  />
                ))}
              </div>
              <button
                onClick={() => setStep(s => Math.min(6, s + 1))}
                disabled={step >= 6}
                className="text-xs font-mono px-3 py-1 border border-grove text-grove rounded hover:bg-grove hover:text-white transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
              >
                next step →
              </button>
            </div>
          </div>

          {/* Mask visualization */}
          <AttentionMaskViz step={step} />

          {/* Collapsible insight boxes at step 6 */}
          {step >= 6 && (
            <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}>
              <button
                onClick={() => setShowInsights(v => !v)}
                className="text-xs font-mono text-grove hover:text-ink transition-colors"
              >
                {showInsights ? '▾ hide key properties' : '▸ show key properties'}
              </button>
              {showInsights && (
                <div className="grid grid-cols-2 gap-3 text-xs mt-2">
                  <div className="border-l-2 border-grove pl-3 py-1">
                    <span className="font-semibold text-grove">Key property:</span>
                    <span className="text-ink-soft ml-1">
                      Each node's attention pattern is exactly the tokens on its ancestor path —
                      making tree-attention a valid verification of speculative paths.
                    </span>
                  </div>
                  <div className="border-l-2 border-depth-1 pl-3 py-1">
                    <span className="font-semibold text-depth-1">Position IDs:</span>
                    <span className="text-ink-soft ml-1">
                      Depth-d nodes use position ID = prefix_len + d − 1.
                      All depth-2 nodes share P+1, depth-3 share P+2 — same slot, different branches.
                    </span>
                  </div>
                </div>
              )}
            </motion.div>
          )}

          <div className="text-xs font-mono text-ink-soft text-center">
            scroll or click next step ({step + 1}/7)
          </div>
        </div>
      </div>
    </section>
  )
}
