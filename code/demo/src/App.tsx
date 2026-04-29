import { useState, useEffect } from 'react'
import NavDots from './components/layout/NavDots'
import HeroSection from './components/sections/HeroSection'
import ArchitectureSection from './components/sections/ArchitectureSection'
import ProposalSection from './components/sections/ProposalSection'
import TreeSection from './components/sections/TreeSection'
import AttentionMaskSection from './components/sections/AttentionMaskSection'
import AcceptSection from './components/sections/AcceptSection'
import ResultsSection from './components/sections/ResultsSection'

export default function App() {
  const [activeSection, setActiveSection] = useState(0)

  useEffect(() => {
    const handleScroll = () => {
      const mid = window.innerHeight / 2
      for (let i = 6; i >= 0; i--) {
        const el = document.getElementById(`section-${i}`)
        if (el) {
          const rect = el.getBoundingClientRect()
          if (rect.top <= mid) {
            setActiveSection(i)
            return
          }
        }
      }
    }
    window.addEventListener('scroll', handleScroll, { passive: true })
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  return (
    <div className="bg-parchment text-ink font-sans">
      <NavDots activeSection={activeSection} />
      <HeroSection />
      <ArchitectureSection />
      <ProposalSection />
      <TreeSection />
      <AttentionMaskSection />
      <AcceptSection />
      <ResultsSection />
    </div>
  )
}

