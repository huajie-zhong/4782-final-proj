import { useRef, useState, useEffect } from 'react'
import TokenStreamViz from '../viz/TokenStreamViz'

const PRIMARY_URL = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '')
const FALLBACK_URL = (import.meta.env.VITE_FALLBACK_URL ?? '').replace(/\/$/, '')

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

type DoneEvent = {
  total_tokens: number
  avg_acceptance: number
  tps: number
  elapsed: number
}

const EXAMPLE_PROMPTS = [
  'Write a Python function to compute Fibonacci numbers.',
  'Explain the theory of relativity in simple terms.',
  'What are the benefits of eating healthy food?',
  'Compose a short poem about the moon.',
]

const BUDGETS = [64, 128, 256, 512]

type StreamState = {
  steps: StepEvent[]
  isStreaming: boolean
  done: DoneEvent | null
  error: string | null
  backendLabel: 'GPU' | 'CPU' | null
  status: 'idle' | 'queued' | 'generating' | 'done' | 'error'
}

const INITIAL_STATE: StreamState = {
  steps: [],
  isStreaming: false,
  done: null,
  error: null,
  backendLabel: null,
  status: 'idle'
}

export default function LiveDemoSection() {
  const [prompt, setPrompt] = useState(EXAMPLE_PROMPTS[0])
  const [maxTokens, setMaxTokens] = useState(64)
  const [mode, setMode] = useState<'base' | 'medusa' | 'compare'>('compare')
  
  const [medusa, setMedusa] = useState<StreamState>(INITIAL_STATE)
  const [base, setBase] = useState<StreamState>(INITIAL_STATE)

  const medusaEsRef = useRef<EventSource | null>(null)
  const baseEsRef = useRef<EventSource | null>(null)
  const medusaGotDataRef = useRef(false)
  const baseGotDataRef = useRef(false)

  useEffect(() => () => { 
    medusaEsRef.current?.close()
    baseEsRef.current?.close()
  }, [])

  const openStream = (
    baseUrl: string,
    isFallback: boolean,
    p: string,
    tokens: number,
    inferenceMode: 'base' | 'medusa',
    updateState: React.Dispatch<React.SetStateAction<StreamState>>,
    esRef: React.MutableRefObject<EventSource | null>,
    gotDataRef: React.MutableRefObject<boolean>
  ) => {
    esRef.current?.close()
    gotDataRef.current = false

    const params = new URLSearchParams({
      prompt: p,
      max_new_tokens: String(tokens),
      mode: inferenceMode,
    })
    const es = new EventSource(`${baseUrl}/generate?${params}`)
    esRef.current = es

    es.onmessage = (e) => {
      gotDataRef.current = true
      let event: Record<string, unknown>
      try { event = JSON.parse(e.data) } catch { return }

      if (event.type === 'tokens') {
        updateState(prev => ({
          ...prev,
          backendLabel: isFallback ? 'CPU' : 'GPU',
          status: 'generating',
          steps: [...prev.steps, event as unknown as StepEvent]
        }))
      } else if (event.type === 'done') {
        updateState(prev => ({
          ...prev,
          done: event as unknown as DoneEvent,
          isStreaming: false,
          status: 'done'
        }))
        es.close()
      } else if (event.type === 'error') {
        updateState(prev => ({
          ...prev,
          error: (event.message as string) ?? 'Inference error',
          isStreaming: false,
          status: 'error'
        }))
        es.close()
      }
    }

    es.onerror = () => {
      es.close()
      if (!gotDataRef.current && !isFallback && FALLBACK_URL) {
        openStream(FALLBACK_URL, true, p, tokens, inferenceMode, updateState, esRef, gotDataRef)
      } else if (!gotDataRef.current) {
        updateState(prev => ({
          ...prev,
          error: 'Connection failed',
          isStreaming: false,
          status: 'error'
        }))
      }
    }
  }

  const handleGenerate = () => {
    medusaEsRef.current?.close()
    baseEsRef.current?.close()
    
    if (mode === 'medusa' || mode === 'compare') {
      setMedusa({ ...INITIAL_STATE, isStreaming: true, status: 'queued' })
      openStream(PRIMARY_URL, false, prompt, maxTokens, 'medusa', setMedusa, medusaEsRef, medusaGotDataRef)
    } else {
      setMedusa(INITIAL_STATE)
    }

    if (mode === 'base' || mode === 'compare') {
      setBase({ ...INITIAL_STATE, isStreaming: true, status: 'queued' })
      openStream(PRIMARY_URL, false, prompt, maxTokens, 'base', setBase, baseEsRef, baseGotDataRef)
    } else {
      setBase(INITIAL_STATE)
    }
  }

  const handleReset = () => {
    medusaEsRef.current?.close()
    baseEsRef.current?.close()
    setMedusa(INITIAL_STATE)
    setBase(INITIAL_STATE)
  }

  const renderStreamBox = (state: StreamState, title: string, subtitle: string, accentColor: string, defaultRate = 0) => {
    const lastStep = state.steps[state.steps.length - 1]
    const liveTps = state.done?.tps ?? lastStep?.tps ?? 0
    const liveRate = (state.done?.avg_acceptance || lastStep?.avg_rate) ?? defaultRate
    const liveTotal = state.done?.total_tokens ?? lastStep?.total ?? 0

    return (
      <div className={`flex-1 min-w-[300px] bg-white border ${state.status === 'generating' ? `border-${accentColor}` : 'border-birch'} rounded-lg p-5 flex flex-col gap-3 transition-colors duration-500`}>
        <div className="flex items-center justify-between">
          <div>
            <h3 className={`text-xs font-mono font-bold uppercase tracking-wider text-${accentColor}`}>{title}</h3>
            <p className="text-[10px] text-ink-soft font-mono">{subtitle}</p>
          </div>
          {state.backendLabel && (
            <span className={`text-[9px] font-mono px-1.5 py-0.5 rounded border ${
              state.backendLabel === 'GPU' ? 'bg-emerald-50 text-emerald-700 border-emerald-200' : 'bg-amber-50 text-amber-700 border-amber-200'
            }`}>
              {state.backendLabel}
            </span>
          )}
        </div>

        <div className="flex-1 min-h-32 p-3 bg-parchment/30 rounded border border-birch/50 overflow-auto">
          {state.status === 'queued' && <p className="text-xs font-mono text-ink-soft animate-pulse">Connecting...</p>}
          <TokenStreamViz steps={state.steps} isStreaming={state.isStreaming} />
          {state.error && <p className="mt-2 text-[10px] text-red-600 font-mono">{state.error}</p>}
        </div>

        <div className="grid grid-cols-3 gap-2 pt-2 border-t border-birch/30">
          <div className="text-center">
            <div className={`text-sm font-bold font-mono text-${accentColor}`}>{liveTotal}</div>
            <div className="text-[9px] text-ink-soft uppercase">Tokens</div>
          </div>
          <div className="text-center">
            <div className={`text-sm font-bold font-mono text-${accentColor}`}>{liveRate.toFixed(1)}</div>
            <div className="text-[9px] text-ink-soft uppercase">Tok/Pass</div>
          </div>
          <div className="text-center">
            <div className={`text-sm font-bold font-mono text-${accentColor}`}>{liveTps.toFixed(1)}</div>
            <div className="text-[9px] text-ink-soft uppercase">TPS</div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <section id="live-demo" className="min-h-screen flex flex-col justify-center border-t border-birch bg-parchment">
      <div className="max-w-6xl mx-auto w-full px-4 py-16 flex flex-col gap-8">

        {/* Header */}
        <div className="text-center">
          <div className="font-mono text-[10px] md:text-xs text-grove border-b border-grove pb-0.5 mb-4 inline-block tracking-widest uppercase">
            §07 — Live Inference
          </div>
          <h2 className="text-2xl md:text-3xl font-bold text-ink">Base vs. Medusa</h2>
          <p className="mt-2 text-sm text-ink-soft max-w-xl mx-auto">
            Compare standard autoregressive generation with Medusa speculative decoding.
            Watch the green highlights — those are "free" tokens predicted by Medusa heads.
          </p>
        </div>

        {/* Controls */}
        <div className="bg-white border border-birch rounded-lg p-5 flex flex-col gap-4">
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            maxLength={300}
            rows={2}
            className="w-full resize-none border border-birch rounded p-3 text-sm font-sans text-ink bg-parchment focus:outline-none focus:border-grove"
            placeholder="Enter a prompt…"
          />

          <div className="flex flex-wrap items-center gap-6">
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-mono text-ink-soft uppercase tracking-wider">Mode</span>
              {(['base', 'medusa', 'compare'] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  className={`px-3 py-1 rounded text-xs font-mono border transition-all ${
                    mode === m ? 'bg-grove text-white border-grove shadow-sm' : 'border-birch text-ink-soft hover:border-ink-soft'
                  }`}
                >
                  {m === 'base' ? 'Base (1 Head)' : m === 'medusa' ? 'Head (Medusa)' : 'Compare Both'}
                </button>
              ))}
            </div>

            <div className="flex items-center gap-2">
              <span className="text-[11px] font-mono text-ink-soft uppercase tracking-wider">Length</span>
              {BUDGETS.map((b) => (
                <button
                  key={b}
                  onClick={() => setMaxTokens(b)}
                  className={`px-2 py-1 rounded text-[10px] font-mono border transition-colors ${
                    maxTokens === b ? 'bg-ink text-white border-ink' : 'border-birch text-ink-soft hover:border-ink-soft'
                  }`}
                >
                  {b}
                </button>
              ))}
            </div>

            <div className="flex items-center gap-3 ml-auto">
              <button onClick={handleReset} className="text-xs font-mono text-ink-soft hover:text-ink transition-colors">Reset</button>
              <button
                onClick={handleGenerate}
                disabled={medusa.isStreaming || base.isStreaming || !prompt.trim()}
                className="px-5 py-2 rounded bg-grove text-white text-xs font-mono tracking-wide disabled:opacity-40 hover:opacity-80 transition-all shadow-md active:scale-95"
              >
                {medusa.isStreaming || base.isStreaming ? 'Running Inference...' : 'Generate →'}
              </button>
            </div>
          </div>
        </div>

        {/* Output Area */}
        <div className="flex flex-wrap gap-6 items-stretch">
          {(mode === 'base' || mode === 'compare') && renderStreamBox(base, "Base (1 LM Head)", "Standard 1-token-at-a-time", "blue-600", 1)}
          {(mode === 'medusa' || mode === 'compare') && renderStreamBox(medusa, "Medusa Head", "Speculative multiple tokens", "grove")}
        </div>

        {/* Legend / Hint */}
        <div className="flex flex-col items-center gap-4 mt-4">
          <div className="flex flex-wrap justify-center gap-8 text-[11px] font-mono">
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 rounded-sm bg-blue-50 border-b-2 border-blue-500 text-blue-800">Normal Token</span>
              <span className="text-ink-soft">Standard pass (Base model)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 rounded-sm bg-green-50 border-b-2 border-green-500 text-green-800">Extra Token</span>
              <span className="text-ink-soft">Speculated & verified (Medusa)</span>
            </div>
          </div>
          <p className="text-[10px] text-ink-soft font-mono max-w-lg text-center opacity-70">
            Hint: Medusa heads predict multiple future tokens in parallel. If verified, they are shown in <span className="text-green-600 font-bold">green</span>. 
            The standard model output is shown in <span className="text-blue-600 font-bold">blue</span>.
          </p>
        </div>

        <p className="text-center text-[10px] text-ink-soft font-mono opacity-60">
          Backend: TinyLlama-1.1B + Medusa Heads on NVIDIA T4 (Modal) · Cold start may take ~30s
        </p>
      </div>
    </section>
  )
}
