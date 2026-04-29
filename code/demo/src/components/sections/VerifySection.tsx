import { motion } from 'framer-motion'
import { useScrollSection } from '../../hooks/useScrollSection'
import VerifyPassViz from '../viz/VerifyPassViz'
import { fadeUp, staggerContainer } from '../../lib/variants'

export default function VerifySection() {
  const { ref, inView } = useScrollSection(0.15)

  return (
    <section id="section-5" className="min-h-screen flex flex-col justify-center px-8 py-20 bg-white border-t border-birch">
      <div className="max-w-5xl mx-auto w-full">
        <motion.div
          ref={ref as React.Ref<HTMLDivElement>}
          variants={staggerContainer}
          initial="hidden"
          animate={inView ? 'visible' : 'hidden'}
          className="flex flex-col gap-8"
        >
          <motion.div variants={fadeUp} className="text-center">
            <div className="font-mono text-xs text-grove border-b border-grove pb-0.5 mb-4 inline-block tracking-widest uppercase">
              §05 — Verify Pass
            </div>
            <h2 className="text-3xl font-bold text-ink">
              One forward pass. 64 candidates verified.
            </h2>
            <p className="mt-3 text-ink-soft max-w-2xl mx-auto">
              All 64 candidate tokens are fed to the backbone in a single forward pass with the custom
              tree-attention mask and depth-based position IDs. The model produces logits for every
              candidate position simultaneously.
            </p>
          </motion.div>

          <motion.div variants={fadeUp}>
            <VerifyPassViz inView={inView} />
          </motion.div>

          <motion.div variants={fadeUp} className="grid md:grid-cols-3 gap-4 text-xs">
            {[
              {
                code: 'verify_input = [lm_token, tree_tokens...]\nshape: (1, 1 + 64)',
                desc: '1 LM token + 64 tree candidates fed together',
              },
              {
                code: 'verify_logits: (1, 65, vocab_size)\none logit vector per position',
                desc: 'Logits at position i predict the next token after token i',
              },
              {
                code: 'past_kv_verify: 65 positions\nsurgery will prune rejected ones',
                desc: 'KV cache grows to 65 entries; most will be discarded',
              },
            ].map(({ code, desc }, i) => (
              <div key={i} className="bg-parchment border border-birch rounded p-4">
                <div className="font-mono text-xs bg-white rounded p-2 mb-2 text-ink-soft whitespace-pre border border-birch">{code}</div>
                <div className="text-ink-soft">{desc}</div>
              </div>
            ))}
          </motion.div>

          <motion.div variants={fadeUp} className="border-l-2 border-grove pl-4 text-sm text-ink-soft">
            <span className="font-semibold text-grove">The critical insight:</span>{' '}
            Because the backbone uses the tree-attention mask, each candidate position
            effectively sees only the tokens along its ancestor path — exactly as if the sequence
            were generated autoregressively. The verify pass produces the same logits as 64 separate
            autoregressive passes would, but in a single batched operation.
          </motion.div>
        </motion.div>
      </div>
    </section>
  )
}
