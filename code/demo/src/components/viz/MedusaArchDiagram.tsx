import { motion } from 'framer-motion'
import { getDepthColor } from '../../data/tree'

const HEAD_LABELS = [
  { k: 0, offset: '+2', color: getDepthColor(1) },
  { k: 1, offset: '+3', color: getDepthColor(2) },
  { k: 2, offset: '+4', color: getDepthColor(3) },
  { k: 3, offset: '+5', color: getDepthColor(4) },
]

const BLOCKS = 6

type Props = { inView: boolean }

export default function MedusaArchDiagram({ inView }: Props) {
  return (
    <div className="flex flex-col items-center gap-0 select-none">
      {/* Transformer blocks */}
      <div className="flex flex-col items-center gap-1">
        {Array.from({ length: BLOCKS }).map((_, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, x: -20 }}
            animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ delay: i * 0.07, duration: 0.4 }}
            className="w-48 h-9 rounded bg-parchment border border-birch flex items-center justify-center"
          >
            <span className="text-xs font-mono text-ink-soft">
              {i === BLOCKS - 1 ? 'Transformer Block (final)' : `Transformer Block ${i + 1}`}
            </span>
          </motion.div>
        ))}
      </div>

      {/* Arrow down */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={inView ? { opacity: 1 } : {}}
        transition={{ delay: 0.5 }}
        className="flex flex-col items-center my-1"
      >
        <div className="w-0.5 h-4 bg-birch" />
        <div className="text-ink-soft text-xs font-mono">hidden states h</div>
        <div className="w-0.5 h-4 bg-birch" />
        <div className="w-0 h-0 border-l-[6px] border-r-[6px] border-t-[8px] border-l-transparent border-r-transparent border-t-birch" />
      </motion.div>

      {/* Split: LM Head + Medusa Heads */}
      <div className="flex items-start gap-4">
        {/* Original LM Head */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ delay: 0.65, duration: 0.4 }}
          className="flex flex-col items-center"
        >
          <div className="w-32 h-10 rounded border-2 border-ink bg-white flex items-center justify-center">
            <span className="text-xs font-semibold font-mono text-ink">LM Head</span>
          </div>
          <div className="w-0.5 h-4 bg-birch" />
          <div className="text-xs font-mono text-ink-soft px-2 py-0.5 bg-parchment rounded border border-birch">
            next token logits
          </div>
        </motion.div>

        {/* Medusa Heads */}
        <div className="flex flex-col items-center gap-2">
          <div className="text-xs font-mono text-ink-soft mb-1">MEDUSA Heads (frozen backbone)</div>
          <div className="flex gap-2">
            {HEAD_LABELS.map(({ k, offset, color }, i) => (
              <motion.div
                key={k}
                initial={{ opacity: 0, scale: 0.7 }}
                animate={inView ? { opacity: 1, scale: 1 } : {}}
                transition={{ delay: 0.75 + i * 0.12, type: 'spring', stiffness: 280, damping: 22 }}
                className="flex flex-col items-center gap-1"
              >
                <div
                  className="w-24 rounded border-2 p-2 text-center"
                  style={{ borderColor: color, backgroundColor: `${color}18` }}
                >
                  <div className="text-xs font-bold font-mono" style={{ color }}>Head {k}</div>
                  <div className="text-[10px] font-mono text-ink-soft mt-0.5">t{offset}</div>
                </div>
                <div className="w-0.5 h-3" style={{ backgroundColor: color }} />
                {/* Formula */}
                <div
                  className="text-[9px] font-mono px-1.5 py-1 rounded border text-center leading-tight max-w-[96px]"
                  style={{ borderColor: `${color}60`, backgroundColor: `${color}0d`, color: '#6B6460' }}
                >
                  W₁<span className="opacity-50">(0)</span>→SiLU<br/>→+h→W₂<span className="opacity-50">(LM)</span>
                </div>
                {/* Zero-init badge */}
                {i === 0 && (
                  <motion.div
                    animate={{ opacity: [1, 0.4, 1] }}
                    transition={{ repeat: Infinity, duration: 2 }}
                    className="text-[9px] px-1.5 py-0.5 rounded border font-semibold font-mono"
                    style={{ borderColor: color, color, backgroundColor: `${color}15` }}
                  >
                    W₁ zeroed
                  </motion.div>
                )}
              </motion.div>
            ))}
          </div>
          <div className="text-[10px] font-mono text-ink-soft text-center mt-1">
            Each head predicts tokens at a different future position
          </div>
        </div>
      </div>
    </div>
  )
}
