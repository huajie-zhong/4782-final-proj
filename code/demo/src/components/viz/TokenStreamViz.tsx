import { motion, AnimatePresence } from 'framer-motion'

type Token = {
  text: string
  spec: boolean
}

type StepEvent = {
  step: number
  tokens: Token[]
  accepted: number
  avg_rate: number
  tps: number
  total: number
}

type Props = {
  steps: StepEvent[]
  isStreaming: boolean
}

const NORMAL_COLOR = { bg: '#eff6ff', border: '#3b82f6', text: '#1e3a8a' } // blue
const SPEC_COLOR = { bg: '#f0fdf4', border: '#22c55e', text: '#166534' }   // green

export default function TokenStreamViz({ steps, isStreaming }: Props) {
  if (steps.length === 0) {
    return (
      <div className="min-h-24 flex items-center justify-center text-ink-soft text-sm font-mono">
        {isStreaming ? 'Generating…' : 'Output will appear here'}
      </div>
    )
  }

  return (
    <div className="font-mono text-sm leading-relaxed break-words">
      <AnimatePresence initial={false}>
        {steps.map((s) => {
          if (s.tokens.length === 0) return null
          
          return (
            <span key={s.step} className="inline whitespace-pre-wrap">
              {s.tokens.map((t, i) => {
                const color = t.spec ? SPEC_COLOR : NORMAL_COLOR
                return (
                  <motion.span
                    key={`${s.step}-${i}`}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ 
                      duration: 0.15, 
                      delay: i * 0.05 // Stagger within step
                    }}
                    style={{
                      backgroundColor: color.bg,
                      borderBottom: `2px solid ${color.border}`,
                      color: color.text,
                      padding: '0 1px',
                      marginRight: '0px',
                      borderRadius: '2px',
                    }}
                    className="inline relative group"
                    title={t.spec ? "Speculative token (MEDUSA)" : "Base model token"}
                  >
                    {t.text}
                  </motion.span>
                )
              })}
            </span>
          )
        })}
      </AnimatePresence>
      {isStreaming && (
        <motion.span 
          animate={{ opacity: [1, 0, 1] }}
          transition={{ repeat: Infinity, duration: 0.8 }}
          className="inline-block w-2 h-4 bg-grove ml-0.5 align-middle" 
        />
      )}
    </div>
  )
}
