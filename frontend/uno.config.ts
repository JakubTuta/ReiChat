import {
  defineConfig,
  presetAttributify,
  presetIcons,
  presetTypography,
  presetUno,
  transformerDirectives,
  transformerVariantGroup,
} from 'unocss'

export default defineConfig({
  theme: {
    colors: {
      'paper': 'var(--color-paper)',
      'card': 'var(--color-card)',
      'soft': 'var(--color-soft)',
      'soft-2': 'var(--color-soft-2)',
      'line': 'var(--color-line)',
      'line-2': 'var(--color-line-2)',
      'line-dash': 'var(--color-line-dash)',
      'ink': 'var(--color-ink)',
      'ink-2': 'var(--color-ink-2)',
      'ink-3': 'var(--color-ink-3)',
      'mint': 'var(--color-mint)',
      'mint-btn': 'var(--color-mint-btn)',
      'mint-btn-hover': 'var(--color-mint-btn-hover)',
      'mint-tint': 'var(--color-mint-tint)',
      'mint-deep': 'var(--color-mint-deep)',
      'mint-ink': 'var(--color-mint-ink)',
      'mint-border': 'var(--color-mint-border)',
      'grey-dot': 'var(--color-grey-dot)',
      'amber': 'var(--color-amber)',
      'red': 'var(--color-red)',
    },
    fontFamily: {
      sans: 'var(--font-sans)',
      mono: 'var(--font-mono)',
    },
  },
  shortcuts: [
    ['chip-mint', 'inline-flex items-center gap-1.5 rounded-full bg-mint-tint px-2.5 py-1 text-[11.5px]'],
    ['hairline', 'border border-line'],
  ],
  presets: [
    presetUno(),
    presetAttributify(),
    presetIcons({
      scale: 1.2,
    }),
    presetTypography(),
  ],
  transformers: [
    transformerDirectives(),
    transformerVariantGroup(),
  ],
})
