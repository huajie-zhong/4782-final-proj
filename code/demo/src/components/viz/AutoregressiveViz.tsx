import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { AUTOREGRESSIVE_TOKENS, MEDUSA_TOKEN_GROUPS } from '../../data/tokens'

type Props = { active: boolean }

export default function AutoregressiveViz({ active }: Props) {
  const [arStep, setArStep] = useState(0)
  const [medusaStep, setMedusaStep] = useState(0)

  useEffect(() => {
    if (!active) return
    setArStep(0)
    setMedusaStep(0)

    const arInterval = setInterval(() => {
      setArStep(prev => (prev < AUTOREGRESSIVE_TOKENS.length ? prev + 1 : prev))
    }, 450)

    const medusaInterval = setInterval(() => {
      setMedusaStep(prev => (prev < MEDUSA_TOKEN_GROUPS.length ? prev + 1 : prev))
    }, 500)

    return () => {
      clearInterval(arInterval)
      clearInterval(medusaInterval)
    }
  }, [active])

  const arTokens = AUTOREGRESSIVE_TOKENS.slice(0, arStep)
  const medusaTokens = MEDUSA_TOKEN_GROUPS.slice(0, medusaStep).flat()

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-6 w-full">
      {/* Autoregressive side */}
      <div className="bg-white border border-birch rounded p-4 md:p-5">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-2 h-2 rounded-sm bg-red-400 animate-pulse" />
          <span className="text-[10px] md:text-xs font-mono text-ink-soft uppercase tracking-wider">
            Autoregressive
          </span>
          <span className="ml-auto text-[10px] md:text-xs text-ink-soft font-mono whitespace-nowrap">
            {arStep} token{arStep !== 1 ? 's' : ''}
          </span>
        </div>
        <div className="font-mono text-xs md:text-sm text-ink min-h-[60px] md:min-h-[80px] flex flex-wrap gap-1 items-start">
          {arTokens.map((tok, i) => (
            <motion.span
              key={i}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2 }}
              className="px-1.5 py-0.5 bg-parchment border border-birch rounded"
            >
              {tok}
            </motion.span>
          ))}
          {active && arStep < AUTOREGRESSIVE_TOKENS.length && (
            <motion.span
              animate={{ opacity: [1, 0, 1] }}
              transition={{ repeat: Infinity, duration: 0.8 }}
              className="inline-block w-1.5 h-4 bg-birch rounded-sm"
            />
          )}
        </div>
        <div className="mt-3 flex items-center gap-2 text-[10px] md:text-xs text-ink-soft font-mono">
          <span>1 token / pass</span>
          <span className="ml-auto">~33 tok/s</span>
        </div>
      </div>

      {/* MEDUSA side */}
      <div className="bg-grove-light border border-grove rounded p-4 md:p-5">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-2 h-2 rounded-sm bg-grove animate-pulse" />
          <span className="text-[10px] md:text-xs font-mono text-grove uppercase tracking-wider">
            MEDUSA
          </span>
          <span className="ml-auto text-[10px] md:text-xs text-grove font-mono whitespace-nowrap">
            {medusaTokens.length} token{medusaTokens.length !== 1 ? 's' : ''}
          </span>
        </div>
        <div className="font-mono text-xs md:text-sm text-ink min-h-[60px] md:min-h-[80px] flex flex-wrap gap-1 items-start">
          <AnimatePresence>
            {MEDUSA_TOKEN_GROUPS.slice(0, medusaStep).map((group, gi) => (
              group.map((tok, ti) => (
                <motion.span
                  key={`${gi}-${ti}`}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.15, delay: ti * 0.06 }}
                  className="px-1.5 py-0.5 bg-white border border-grove rounded text-grove"
                >
                  {tok}
                </motion.span>
              ))
            ))}
          </AnimatePresence>
          {active && medusaStep < MEDUSA_TOKEN_GROUPS.length && (
            <motion.span
              animate={{ opacity: [1, 0, 1] }}
              transition={{ repeat: Infinity, duration: 0.8 }}
              className="inline-block w-1.5 h-4 bg-grove rounded-sm"
            />
          )}
        </div>
        <div className="mt-3 flex items-center gap-2 text-[10px] md:text-xs text-grove font-mono">
          <span>~3 tokens / pass</span>
          <span className="ml-auto font-semibold">~72 tok/s</span>
        </div>
      </div>
    </div>
  )
}
