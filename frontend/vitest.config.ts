import { fileURLToPath } from 'node:url'
import { defineVitestProject } from '@nuxt/test-utils/config'
import { defineConfig } from 'vitest/config'

const root = fileURLToPath(new URL('.', import.meta.url))

export default defineConfig({
  test: {
    projects: [
      // Pure logic and stores: fast, no Nuxt runtime.
      {
        resolve: {
          alias: { '~': root, '@': root },
        },
        test: {
          name: 'unit',
          environment: 'happy-dom',
          include: ['tests/unit/**/*.spec.ts'],
        },
      },
      // Components: mounted inside the real Nuxt app, so its plugins (Vuetify,
      // Pinia), auto-imports and layouts are live. Not the place to test
      // sanitising: the fake DOM can't (see tests/browser/security.spec.ts).
      await defineVitestProject({
        test: {
          name: 'nuxt',
          environment: 'nuxt',
          include: ['tests/nuxt/**/*.spec.ts'],
          setupFiles: ['tests/nuxt/setup.ts'],
          // plugins/vuetify.ts reads its colours from main.css as raw text, and
          // Vitest blanks CSS imports unless told to process them.
          css: true,
          server: { deps: { inline: ['vuetify'] } },
        },
      }),
    ],
  },
})
