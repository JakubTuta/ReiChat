import type { Page } from '@playwright/test'
import { expect, test } from '@playwright/test'
import { mockBackend } from './support/mockBackend'
import { expectLocatorScreenshot, expectPageScreenshot } from './support/screenshot'

async function openModels(page: Page, query = '') {
  const backend = await mockBackend(page)

  await page.goto(`/models${query}`)
  await expect(page.getByRole('heading', { name: 'Models' })).toBeVisible()

  return backend
}

function card(page: Page, name: string) {
  return page.locator('.model-card').filter({ has: page.getByText(name, { exact: true }) })
}

test('every card state: running, stopped, downloading, incomplete, restarting', async ({ page }) => {
  const backend = await openModels(page)

  // Models with a container sort first, so all five states fit one screen.
  await expectPageScreenshot(page, 'models-card-states.png', { fullPage: true })
  expect(backend.unmocked).toEqual([])
})

test('the advanced run-manually panel expands to the docker commands', async ({ page }) => {
  await openModels(page)

  await card(page, 'deepseek-r1').getByText('advanced: run manually').click()
  await expect(card(page, 'deepseek-r1').getByText('docker stop deepseek-r1_1.5b')).toBeVisible()

  // Vuetify animates the panel open by pinning an inline pixel height and
  // clears it when done; a screenshot taken before that is a rounded pixel
  // taller than the settled card.
  const panelText = card(page, 'deepseek-r1').locator('.v-expansion-panel-text')
  await expect.poll(() => panelText.evaluate(el => (el as HTMLElement).style.height)).toBe('')

  await expectLocatorScreenshot(card(page, 'deepseek-r1'), 'models-advanced-open.png')
})

test('removing a model asks for confirmation first', async ({ page }) => {
  await openModels(page)

  await card(page, 'llama3.1').getByRole('button', { name: 'Remove' }).click()
  await expect(page.getByText('This cannot be undone.')).toBeVisible()

  await expectPageScreenshot(page, 'models-remove-confirm.png')
})

test('add a model by name dialog', async ({ page }) => {
  await openModels(page)

  await page.getByRole('button', { name: 'Add by name' }).click()
  await expect(page.getByRole('dialog')).toBeVisible()

  await expectPageScreenshot(page, 'models-add-by-name.png')
})

test('change where models run dialog', async ({ page }) => {
  await openModels(page)

  await page.getByRole('button', { name: /Managed by ReiChat/ }).click()
  await expect(page.getByRole('dialog')).toBeVisible()

  await expectPageScreenshot(page, 'models-connection.png')
})

test('refresh model list dialog', async ({ page }) => {
  await openModels(page)

  await page.getByRole('button', { name: 'Refresh model list' }).click()
  await expect(page.getByRole('dialog')).toBeVisible()

  await expectPageScreenshot(page, 'models-refresh-list.png')
})

test('filter pills and search narrow the catalogue', async ({ page }) => {
  await openModels(page, '?filter=stoppedContainers&sort=nameAlphabetically&shell=cmd&canProcessImages=true')

  await expectPageScreenshot(page, 'models-filtered-empty.png')
})

test('a search that matches one model', async ({ page }) => {
  await openModels(page, '?search=llama')

  await expect(card(page, 'llama3.1')).toBeVisible()
  await expectPageScreenshot(page, 'models-search.png', { fullPage: true })
})

test('no models loaded yet shows the empty state', async ({ page }) => {
  await mockBackend(page, { aiModels: [], containers: [], pullProgress: {}, loadedModels: [] })

  await page.goto('/models')
  await expect(page.getByText('No models loaded yet')).toBeVisible()

  await expectPageScreenshot(page, 'models-empty.png')
})

test('first run with a catalogue but no chat model installed', async ({ page }) => {
  await mockBackend(page, { containers: [], pullProgress: {}, loadedModels: [] })

  await page.goto('/models')
  await expect(page.getByText('llama3.1', { exact: true }).first()).toBeVisible()

  await expectPageScreenshot(page, 'models-first-run.png', { fullPage: true })
})
