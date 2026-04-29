import type { Config } from 'tailwindcss'

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        'depth-1': '#f59e0b',
        'depth-2': '#10b981',
        'depth-3': '#0ea5e9',
        'depth-4': '#8b5cf6',
        // Academic ink palette
        'ink': '#1A1714',
        'ink-soft': '#6B6460',
        'parchment': '#F7F3EC',
        'birch': '#DDD8CE',
        'grove': '#2A5C45',
        'grove-light': '#EAF2EE',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
} satisfies Config
