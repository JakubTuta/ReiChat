import process from 'node:process'
import { defineConfig, devices } from '@playwright/test'

// Real-browser tests of the whole UI: screenshot regression and behaviour
// that a fake DOM can't prove (sanitising model output, for one). Unlike the
// e2e smoke config this needs no Docker stack: it serves the production build
// and answers every backend call from fixtures (tests/browser/support), so a
// failure can only come from a UI change. Screenshot baselines are per-OS
// (fonts render differently), so regenerate them with
// `npm run test:browser:update` on a new machine.
const PORT = 3100

export default defineConfig({
  testDir: './tests/browser',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  reporter: process.env.CI
    ? 'github'
    : 'list',
  expect: {
    toHaveScreenshot: {
      // Identical builds render identically on one machine, so this only
      // absorbs rasteriser noise. It is deliberately tight: a recoloured
      // label or a 1px shifted border must fail, not slip through.
      threshold: 0.05,
      maxDiffPixels: 50,
    },
  },
  use: {
    baseURL: `http://localhost:${PORT}`,
    locale: 'en-US',
    timezoneId: 'UTC',
    colorScheme: 'light',
    launchOptions: {
      // Chromium antialiases text with subpixel (ClearType) colour fringes
      // or in grayscale depending on whether the layer is opaque yet, so a
      // dialog title screenshotted mid-transition differed from one taken a
      // frame later. Grayscale everywhere makes text render the same
      // regardless of timing.
      args: ['--disable-lcd-text', '--font-render-hinting=none'],
    },
  },
  projects: [
    {
      name: 'desktop',
      use: { ...devices['Desktop Chrome'], viewport: { width: 1280, height: 800 } },
      testIgnore: /.*\.mobile\.spec\.ts/,
    },
    {
      name: 'mobile',
      use: { ...devices['Pixel 7'] },
      testMatch: /.*\.mobile\.spec\.ts/,
    },
  ],
  webServer: {
    // Rebuilt every run so a baseline is never compared against stale code.
    // Set BROWSER_REUSE_SERVER=1 to keep a server you started yourself.
    command: 'npm run build && node .output/server/index.mjs',
    url: `http://localhost:${PORT}`,
    timeout: 300_000,
    reuseExistingServer: process.env.BROWSER_REUSE_SERVER === '1',
    env: { PORT: String(PORT), HOST: '127.0.0.1', NITRO_PORT: String(PORT), NITRO_HOST: '127.0.0.1' },
  },
})
