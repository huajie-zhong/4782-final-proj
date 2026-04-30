const SECTIONS = [
  'The Problem',
  'Architecture',
  'Token Proposals',
  'Candidate Tree',
  'Attention & Verify',
  'Acceptance Walk',
  'Results',
]

type Props = { activeSection: number; range?: [number, number] }

export default function NavDots({ activeSection, range = [0, 6] }: Props) {
  const [start, end] = range
  const sections = SECTIONS.slice(start, end + 1)

  return (
    <nav className="fixed right-4 top-1/2 -translate-y-1/2 z-50 hidden md:flex flex-col items-end gap-0">
      {/* Vertical spine */}
      <div className="absolute right-1.5 top-0 bottom-0 w-px bg-birch" />
      {sections.map((label, relIdx) => {
        const i = relIdx + start
        return (
          <button
            key={i}
            onClick={() => {
              const el = document.getElementById(`section-${i}`)
              el?.scrollIntoView({ behavior: 'smooth' })
            }}
            className="group relative flex items-center gap-2 py-2"
            aria-label={label}
          >
            {/* Label on hover */}
            <span className="opacity-0 group-hover:opacity-100 transition-opacity duration-150 text-[10px] font-mono text-ink-soft whitespace-nowrap pr-1">
              {label}
            </span>
            {/* Tick mark */}
            <span
              className="relative z-10 block rounded-sm transition-all duration-200"
              style={{
                width: activeSection === i ? 14 : 6,
                height: 2,
                backgroundColor: activeSection === i ? '#2A5C45' : '#C4BEB6',
              }}
            />
          </button>
        )
      })}
    </nav>
  )
}
