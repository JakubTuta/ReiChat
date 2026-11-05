import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { oklchToHex } from '~/theme/oklch'
import { buildReichatTheme } from '~/theme/reichat'

// vitest runs from the frontend root; happy-dom's URL can't resolve file paths.
const mainCss = readFileSync(resolve('assets/css/main.css'), 'utf8')

describe('oklchToHex', () => {
  it('maps the extremes to white and black', () => {
    expect(oklchToHex('oklch(1 0 0)')).toBe('#ffffff')
    expect(oklchToHex('oklch(0 0 0)')).toBe('#000000')
  })

  it('matches a colour whose sRGB value was checked by hand', () => {
    expect(oklchToHex('oklch(0.995 0.004 168)')).toBe('#fbfefd')
  })

  it('ignores an alpha channel rather than failing on it', () => {
    expect(oklchToHex('oklch(0.3 0.02 168 / .14)')).toMatch(/^#[\da-f]{6}$/)
  })

  it('rejects anything that is not oklch() so a typo in main.css is loud', () => {
    expect(() => oklchToHex('#ffffff')).toThrow('Not an oklch() colour')
    expect(() => oklchToHex('oklch(banana)')).toThrow('Not an oklch() colour')
  })
})

describe('buildReichatTheme', () => {
  const theme = buildReichatTheme(mainCss)

  it('resolves every Vuetify colour to a hex value Vuetify can parse', () => {
    expect(Object.keys(theme.colors ?? {}).length).toBeGreaterThan(0)

    for (const [name, value] of Object.entries(theme.colors ?? {}))
      expect(value, name).toMatch(/^#[\da-f]{6}$/)
  })

  it('keeps colours that share a token identical', () => {
    expect(theme.colors?.primary).toBe(theme.colors?.['mint-btn'])
    expect(theme.colors?.error).toBe(theme.colors?.red)
    expect(theme.colors?.warning).toBe(theme.colors?.amber)
    expect(theme.colors?.['on-surface']).toBe(theme.colors?.['on-background'])
  })

  it('takes the border colour from the same token the raw CSS uses', () => {
    expect(theme.variables?.['border-color']).toBe(oklchToHex('oklch(0.9 0.012 168)'))
  })

  it('fails loudly, naming the token, when main.css loses one', () => {
    const withoutMint = mainCss.replace(/--color-mint-deep:[^;]+;/, '')

    expect(() => buildReichatTheme(withoutMint)).toThrow('--color-mint-deep')
  })
})
