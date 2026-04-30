import { useEffect } from 'react'
import Navbar from '../components/layout/Navbar'
import HeroSection from '../components/sections/HeroSection'
import LiveDemoSection from '../components/sections/LiveDemoSection'

export default function HomePage() {
  // Scroll to #live-demo when navigating from another page via /#live-demo
  useEffect(() => {
    if (window.location.hash === '#live-demo') {
      setTimeout(() => {
        document.getElementById('live-demo')?.scrollIntoView({ behavior: 'smooth' })
      }, 100)
    }
  }, [])

  return (
    <div className="bg-parchment text-ink font-sans">
      <Navbar />
      <div className="pt-12">
        <HeroSection />
        <LiveDemoSection />
      </div>
    </div>
  )
}
