import { useRef, useEffect, useState } from 'react'
import KVCacheViz from '../viz/KVCacheViz'

const STEPS = ['Before surgery', 'Marking rejected', 'Cache pruned']

export default function KVSurgerySection() {
  const sectionRef = useRef<HTMLDivElement>(null)
  const [animStep, setAnimStep] = useState(0)
  const [manualStep, setManualStep] = useState<number | null>(null)

  useEffect(() => {
    const handleScroll = () => {
      if (manualStep !== null) return
      const section = sectionRef.current
      if (!section) return
      const rect = section.getBoundingClientRect()
      const progress = Math.max(0, Math.min(1, (-rect.top) / (section.offsetHeight - window.innerHeight)))
      setAnimStep(Math.min(2, Math.floor(progress * 3)))
    }
    window.addEventListener('scroll', handleScroll, { passive: true })
    handleScroll()
    return () => window.removeEventListener('scroll', handleScroll)
  }, [manualStep])

  const displayStep = manualStep !== null ? manualStep : animStep

  return (
    <section
      id="section-7"
      ref={sectionRef}
      style={{ height: '350vh' }}
      className="relative border-t border-birch"
    >
      <div className="sticky top-0 h-screen flex flex-col justify-center px-8 py-8 bg-white">
        <div className="max-w-5xl mx-auto w-full flex flex-col gap-6">
          <div>
            <div className="font-mono text-xs text-grove border-b border-grove pb-0.5 mb-3 inline-block tracking-widest uppercase">
              §07 — KV Cache Surgery
            </div>
            <div className="flex items-center gap-3">
              <h2 className="text-2xl font-bold text-ink">
                Prune the cache, keep only the accepted path
              </h2>
              {/* Clickable step pills */}
              <div className="flex gap-1.5 ml-auto">
                {STEPS.map((s, i) => (
                  <button
                    key={i}
                    onClick={() => setManualStep(manualStep === i ? null : i)}
                    className={`text-xs px-2 py-0.5 rounded border transition-colors font-mono ${
                      i === displayStep
                        ? 'bg-grove text-white border-grove'
                        : i < displayStep
                        ? 'bg-grove-light text-grove border-grove'
                        : 'bg-white text-ink-soft border-birch hover:border-grove hover:text-grove'
                    }`}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
            <p className="mt-1 text-ink-soft text-sm max-w-xl">
              After acceptance, the KV cache holds 72 positions (8 prefix + 64 tree nodes).
              We discard all rejected branches via{' '}
              <code className="bg-parchment px-1 rounded text-grove font-mono">index_select</code>{' '}
              on dim=2, keeping only the prefix + accepted path nodes.
            </p>
          </div>

          <KVCacheViz animStep={displayStep} />

          <div className="border-l-2 border-birch pl-4 text-xs text-ink-soft leading-relaxed">
            Without KV surgery, the cache would accumulate all 64 tree nodes every iteration —
            blowing up memory and slowing future decoding. Surgery keeps the cache linear in
            sequence length, just like standard autoregressive decoding.
            {' '}<span className="font-mono text-ink">72 → ~11 positions retained per pass.</span>
          </div>

          <div className="text-xs font-mono text-ink-soft text-center">
            scroll or click a step above ({displayStep + 1}/3)
            {manualStep !== null && (
              <button onClick={() => setManualStep(null)} className="ml-3 text-grove underline">
                resume scroll
              </button>
            )}
          </div>
        </div>
      </div>
    </section>
  )
}
