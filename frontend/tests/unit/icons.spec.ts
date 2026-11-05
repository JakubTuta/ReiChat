import { readdirSync, readFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { appIcons } from '~/theme/icons'

// vitest runs from the frontend root; happy-dom's URL can't resolve file paths.
const root = resolve('.')

function sourceFiles(dir: string): string[] {
  return readdirSync(join(root, dir), { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name)
    if (entry.isDirectory())
      return sourceFiles(path)

    return /\.(?:vue|ts)$/.test(entry.name)
      ? [path]
      : []
  })
}

const sources = [
  ...['pages', 'components', 'composables', 'plugins', 'stores', 'utils'].flatMap(sourceFiles),
  'app.vue',
  'error.vue',
].map(path => ({ path, text: readFileSync(join(root, path), 'utf8') }))

function usedAliases(): Set<string> {
  const used = new Set<string>()
  for (const { text } of sources) {
    for (const [, name] of text.matchAll(/\$(mdi[A-Z]\w*)/g))
      used.add(name)
  }

  return used
}

describe('icon registry', () => {
  it('registers every icon a template uses, or it renders blank', () => {
    const missing = [...usedAliases()].filter(name => !(name in appIcons))

    expect(missing).toEqual([])
  })

  it('has no icon that nothing uses', () => {
    const used = usedAliases()
    const unused = Object.keys(appIcons).filter(name => !used.has(name))

    expect(unused).toEqual([])
  })

  it('has no leftover icon-font class names, which would render blank without the font', () => {
    const stragglers = sources
      .filter(({ text }) => /(?<![\w/])mdi-[a-z]/.test(text))
      .map(({ path }) => path)

    expect(stragglers).toEqual([])
  })
})
