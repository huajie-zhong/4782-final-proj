import { useRef, useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import AcceptanceWalkViz from '../viz/AcceptanceWalkViz'

const ANIM_STEPS = 9

export default function AcceptSection() {
  const sectionRef = useRef<HTMLDivElement>(null)
  const [animStep, setAnimStep] = useState(0)
  const [inView, setInView] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      const section = sectionRef.current
      if (!section) return
      const rect = section.getBoundingClientRect()
      const sectionHeight = section.offsetHeight
      const viewH = window.innerHeight
      setInView(rect.top < viewH && rect.bottom > 0)
      const progress = Math.max(0, Math.min(1, (-rect.top) / (sectionHeight - viewH)))
      setAnimStep(Math.min(ANIM_STEPS - 1, Math.floor(progress * ANIM_STEPS)))
    }
    window.addEventListener('scroll', handleScroll, { passive: true })
    handleScroll()
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  return (
    <section
      id="section-5"
      ref={sectionRef}
      style={{ height: '500vh' }}
      className="relative border-t border-birch"
    >
      <div className="sticky top-0 h-screen flex flex-col justify-center px-8 py-8 bg-parchment">
        <div className="max-w-5xl mx-auto w-full flex flex-col gap-5">
          <div className="flex items-start justify-between">
            <div>
              <div className="font-mono text-xs text-grove border-b border-grove pb-0.5 mb-3 inline-block tracking-widest uppercase">
                §05 — Acceptance Walk
              </div>
              <h2 className="text-2xl font-bold text-ink">
                Walk the tree, collect the longest valid sequence
              </h2>
              <p className="mt-1 text-ink-soft text-sm max-w-xl">
                Starting from the LM token's logits, we walk down the tree: accept a node if the parent's
                predicted argmax matches that node's token. Stop at the first mismatch.
              </p>
            </div>
          </div>

          <AcceptanceWalkViz inView={inView} animStep={animStep} />

          <div className="text-xs font-mono text-ink-soft text-center">
            scroll to animate ({animStep + 1}/{ANIM_STEPS})
          </div>
        </div>
      </div>
    </section>
  )
}
