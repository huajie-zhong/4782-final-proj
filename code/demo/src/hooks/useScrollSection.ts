import { useInView } from 'react-intersection-observer'

export function useScrollSection(threshold = 0.2) {
  const { ref, inView } = useInView({
    threshold,
    triggerOnce: false,
  })
  return { ref, inView }
}

export function useScrollSectionOnce(threshold = 0.2) {
  const { ref, inView } = useInView({
    threshold,
    triggerOnce: true,
  })
  return { ref, inView }
}
