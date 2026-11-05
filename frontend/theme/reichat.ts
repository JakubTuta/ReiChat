import type { ThemeDefinition } from 'vuetify'
import { oklchToHex } from './oklch'

// Vuetify colour name -> the --color-* token in assets/css/main.css it
// mirrors. main.css is the only place a colour value is written down; this
// table is the only place that says which token Vuetify should use, so the
// two can't drift apart. Only colours Vuetify actually reads are listed.
const VUETIFY_COLORS: Record<string, string> = {
  'background': 'paper',
  'on-background': 'ink',
  'surface': 'card',
  'on-surface': 'ink',
  'primary': 'mint-btn',
  'on-primary': 'mint-ink',
  'error': 'red',
  'warning': 'amber',
  'mint': 'mint',
  'mint-btn': 'mint-btn',
  'mint-deep': 'mint-deep',
  'red': 'red',
  'amber': 'amber',
}

const TOKEN_PATTERN = /--color-([\w-]+):\s*(oklch\([^)]*\))\s*;/g

function readTokens(css: string): Map<string, string> {
  const tokens = new Map<string, string>()
  for (const [, name, value] of css.matchAll(TOKEN_PATTERN))
    tokens.set(name, value)

  return tokens
}

export function buildReichatTheme(css: string): ThemeDefinition {
  const tokens = readTokens(css)

  const hex = (token: string): string => {
    const value = tokens.get(token)
    if (!value)
      throw new Error(`main.css has no --color-${token} (an oklch value) but the Vuetify theme needs it.`)

    return oklchToHex(value)
  }

  return {
    dark: false,
    colors: Object.fromEntries(
      Object.entries(VUETIFY_COLORS).map(([name, token]) => [name, hex(token)]),
    ),
    variables: {
      'border-color': hex('line'),
      'border-opacity': 1,
    },
  }
}
