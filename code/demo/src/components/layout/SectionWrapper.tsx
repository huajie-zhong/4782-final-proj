import { useEffect, useRef } from 'react'
import clsx from 'clsx'

type Props = {
  id: string
  index: number
  onVisible: (index: number) => void
  children: React.ReactNode
  className?: string
  fullHeight?: boolean
}

export default function SectionWrapper({ id, index, onVisible, children, className, fullHeight = true }: Props) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) onVisible(index)
      },
      { threshold: 0.3 }
    )
    if (ref.current) observer.observe(ref.current)
    return () => observer.disconnect()
  }, [index, onVisible])

  return (
    <section
      id={id}
      ref={ref}
      className={clsx(
        'relative w-full',
        fullHeight ? 'min-h-screen' : '',
        className
      )}
    >
      {children}
    </section>
  )
}
