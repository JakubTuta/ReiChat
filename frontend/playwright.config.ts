import process from 'node:process'
import { defineConfig, devices } from '@playwright/test'

// This app is a Nuxt SPA + FastAPI + Postgres + Docker stack, not a
// single `npm run dev` process — CI (and local runs) are expected to bring
// up `docker compose` themselves before invoking Playwright, rather than
// Playwright managing a `webServer` here. See tests/e2e/README.md.
export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI
    ? 2
    : 0,
  workers: process.env.CI
    ? 1
    : undefined,
  reporter: process.env.CI
    ? 'github'
    : 'list',
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL ?? 'http://localhost:3000',
    trace: 'on-first-retry',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
})
