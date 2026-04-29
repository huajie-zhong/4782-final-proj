import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, ReferenceLine, Cell,
} from 'recharts'
import { ABLATION_DATA, HEAD_ACCURACY_DATA, BENCHMARK } from '../../data/benchmarks'
import { motion } from 'framer-motion'

const PROMPT_SHORT = ['Python\nsort', 'Relativity\nexplain', 'Healthy\neating', 'Moon\npoem', 'Cookie\nrecipe']

type Props = { inView: boolean }

export default function SpeedupCharts({ inView }: Props) {
  const perPromptData = BENCHMARK.perPromptGreedy.map((r, i) => ({
    name: PROMPT_SHORT[i],
    greedy: +r.acceptanceRate.toFixed(2),
    typical: +BENCHMARK.perPromptTypical[i].acceptanceRate.toFixed(2),
  }))

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.6 }}
      className="grid grid-cols-1 md:grid-cols-3 gap-6"
    >
      {/* Chart 1: Ablation speedup */}
      <div className="bg-white border border-birch rounded p-5">
        <div className="text-sm font-semibold text-ink mb-1 font-mono">Speedup Progression</div>
        <div className="text-xs text-ink-soft mb-4 font-mono">Table 3 ablation (TinyLlama)</div>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={ABLATION_DATA} margin={{ top: 10, right: 10, left: -20, bottom: 40 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="name"
              tick={{ fontSize: 9, fill: '#64748b', fontFamily: 'JetBrains Mono, monospace' }}
              angle={-25}
              textAnchor="end"
              interval={0}
            />
            <YAxis
              domain={[0, 2.5]}
              tick={{ fontSize: 10, fill: '#94a3b8' }}
              tickFormatter={v => `${v}×`}
            />
            <Tooltip formatter={(v: number) => [`${v.toFixed(3)}×`, 'Speedup']} />
            <ReferenceLine y={2.18} stroke="#2A5C45" strokeDasharray="4 2" label={{ value: 'Paper 2.18×', fill: '#2A5C45', fontSize: 9 }} />
            <Bar dataKey="speedup" radius={[3, 3, 0, 0]}>
              {ABLATION_DATA.map((entry, i) => (
                <Cell key={i} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Chart 2: Head accuracy */}
      <div className="bg-white border border-birch rounded p-5">
        <div className="text-sm font-semibold text-ink mb-1 font-mono">Head Accuracy vs. Depth</div>
        <div className="text-xs text-ink-soft mb-4 font-mono">Further predictions = harder</div>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={HEAD_ACCURACY_DATA} margin={{ top: 10, right: 10, left: -20, bottom: 40 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="head"
              tick={{ fontSize: 9, fill: '#64748b', fontFamily: 'JetBrains Mono, monospace' }}
              angle={-20}
              textAnchor="end"
              interval={0}
            />
            <YAxis
              domain={[0, 65]}
              tick={{ fontSize: 10, fill: '#94a3b8' }}
              tickFormatter={v => `${v}%`}
            />
            <Tooltip formatter={(v: number) => [`${v.toFixed(1)}%`, 'Top-1 Accuracy']} />
            <ReferenceLine y={25} stroke="#f97316" strokeDasharray="4 2" label={{ value: '~random', fill: '#f97316', fontSize: 9 }} />
            <Line
              type="monotone"
              dataKey="accuracy"
              stroke="#2A5C45"
              strokeWidth={2.5}
              dot={{ fill: '#2A5C45', r: 5 }}
              activeDot={{ r: 7 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Chart 3: Per-prompt acceptance rates */}
      <div className="bg-white border border-birch rounded p-5">
        <div className="text-sm font-semibold text-ink mb-1 font-mono">Acceptance Rate by Prompt</div>
        <div className="text-xs text-ink-soft mb-4 font-mono">Greedy vs. typical acceptance</div>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={perPromptData} margin={{ top: 10, right: 10, left: -20, bottom: 40 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="name"
              tick={{ fontSize: 9, fill: '#64748b', fontFamily: 'JetBrains Mono, monospace' }}
              angle={-20}
              textAnchor="end"
              interval={0}
            />
            <YAxis
              domain={[0, 2.5]}
              tick={{ fontSize: 10, fill: '#94a3b8' }}
              tickFormatter={v => `${v}×`}
            />
            <Tooltip formatter={(v: number) => [`${v.toFixed(2)} tokens/pass`, '']} />
            <Bar dataKey="greedy" name="Greedy" fill="#0ea5e9" radius={[2, 2, 0, 0]} />
            <Bar dataKey="typical" name="Typical" fill="#8b5cf6" radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  )
}
