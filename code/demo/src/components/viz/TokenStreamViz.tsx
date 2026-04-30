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

const STEP_COLORS = [
  { bg: '#fef3c7', border: '#f59e0b', text: '#92400e' }, // amber
  { bg: '#d1fae5', border: '#10b981', text: '#065f46' }, // emerald
  { bg: '#e0f2fe', border: '#0ea5e9', text: '#0c4a6e' }, // sky
  { bg: '#ede9fe', border: '#8b5cf6', text: '#4c1d95' }, // violet
  { bg: '#fce7f3', border: '#ec4899', text: '#831843' }, // pink
]

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
          const color = STEP_COLORS[Math.abs(s.step) % STEP_COLORS.length]
          if (s.tokens.length === 0) return null
          
          return (
            <span key={s.step} className="inline whitespace-pre-wrap">
              {s.tokens.map((t, i) => (
                <motion.span
                  key={`${s.step}-${i}`}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ 
                    duration: 0.15, 
                    delay: i * 0.05 // Stagger within step
                  }}
                  style={{
                    backgroundColor: t.spec ? '#dcfce7' : color.bg, // green-100 for spec
                    borderBottom: `2px solid ${t.spec ? '#22c55e' : color.border}`, // green-500 for spec
                    color: t.spec ? '#166534' : color.text, // green-800 for spec
                    padding: '0 1px',
                    marginRight: '0px',
                    borderRadius: '2px',
                  }}
                  className="inline relative group"
                  title={t.spec ? "Speculative token (MEDUSA)" : "Base model token"}
                >
                  {t.text}
                </motion.span>
              ))}
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
