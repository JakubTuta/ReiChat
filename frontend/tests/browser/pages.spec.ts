import { expect, test } from '@playwright/test'
import { mockBackend } from './support/mockBackend'
import { expectPageScreenshot } from './support/screenshot'

const pages = [
  { name: 'landing', path: '/', ready: /chat UI for the models/i },
  { name: 'models', path: '/models', ready: /llama3\.1/ },
  { name: 'compare', path: '/compare', ready: /compare/i },
  { name: 'integrations', path: '/integrations', ready: /integrations|web search/i },
  { name: 'data', path: '/data', ready: /your data|backup/i },
] as const

for (const { name, path, ready } of pages) {
  test(`${name} page`, async ({ page }) => {
    const backend = await mockBackend(page)

    await page.goto(path)
    await expect(page.getByText(ready).first()).toBeVisible()

    await expectPageScreenshot(page, `${name}.png`, { fullPage: true })
    expect(backend.unmocked).toEqual([])
    expect(backend.cspViolations).toEqual([])
  })
}

test('chat page with a conversation', async ({ page }) => {
  const backend = await mockBackend(page)

  await page.goto('/chat')
  await expect(page.getByText('Write a Python quicksort function')).toBeVisible()

  await expectPageScreenshot(page, 'chat.png')
  expect(backend.unmocked).toEqual([])
  expect(backend.cspViolations).toEqual([])
})

test('unknown route shows the error page', async ({ page }) => {
  await mockBackend(page)

  await page.goto('/this-page-does-not-exist')

  await expectPageScreenshot(page, 'error.png')
})
