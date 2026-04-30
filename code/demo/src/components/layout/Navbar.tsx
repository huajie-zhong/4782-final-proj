import { Link } from 'react-router-dom'

export default function Navbar() {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-parchment/90 backdrop-blur-sm border-b border-birch">
      <div className="max-w-5xl mx-auto px-6 h-12 flex items-center justify-between">
        <Link to="/" className="font-mono font-bold text-grove text-sm tracking-widest uppercase">
          MEDUSA
        </Link>
        <div className="flex items-center gap-8">
          <Link
            to="/how-it-works"
            className="text-[11px] font-mono text-ink-soft hover:text-ink transition-colors tracking-wider uppercase"
          >
            How It Works
          </Link>
          <a
            href="/#live-demo"
            className="text-[11px] font-mono text-grove hover:opacity-70 transition-opacity tracking-wider uppercase"
          >
            Try Live →
          </a>
        </div>
      </div>
    </nav>
  )
}
