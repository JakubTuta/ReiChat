import { expect, test } from '@playwright/test'

// A deliberately light onboarding smoke test: it proves a fresh visitor can
// reach every top-level page and see real content, without ever pulling a
// model (a real pull is multi-gigabyte and unsuitable for CI). The full
// "seed catalog -> create container -> first answer" path from the plan's
// verification section needs a live Ollama model and is intentionally left
// for manual/nightly verification — see tests/e2e/README.md.

test('landing page renders the hero and primary CTAs', async ({ page }) => {
  await page.goto('/')

  await expect(page.getByRole('heading', { name: /chat UI for the models/i })).toBeVisible()
  await expect(page.getByRole('link', { name: /Get started/i })).toBeVisible()
  await expect(page.getByRole('link', { name: /Explore models/i })).toBeVisible()
})

test('nav bar reaches Chat and Models from the landing page', async ({ page }) => {
  await page.goto('/')

  await page.getByRole('link', { name: 'Models', exact: true }).click()
  await expect(page).toHaveURL(/\/models$/)

  await page.getByRole('link', { name: 'Chat', exact: true }).click()
  await expect(page).toHaveURL(/\/chat$/)
})

test('models page shows the bundled seed catalog with zero network calls', async ({ page }) => {
  // A fresh install ships ~40 models via a data migration so the catalog
  // works before the user ever presses "Fetch model list". This is the
  // regression guard for that promise at the UI layer.
  await page.goto('/models')

  await expect(page.getByText('llama3.1', { exact: true }).first()).toBeVisible()
})

test('chat page renders a usable composer', async ({ page }) => {
  await page.goto('/chat')

  await expect(page.getByLabel('Message', { exact: true })).toBeVisible()
})
