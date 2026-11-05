import { expect, test } from '@playwright/test'
import { mockBackend } from './support/mockBackend'
import { expectPageScreenshot } from './support/screenshot'

test('chat on a phone: drawer closed, then open', async ({ page }) => {
  await mockBackend(page)

  await page.goto('/chat')
  await expect(page.getByText('Write a Python quicksort function')).toBeVisible()
  await expectPageScreenshot(page, 'mobile-chat.png')

  await page.getByRole('button', { name: 'Toggle conversations list' }).click()
  await expect(page.getByText('Weekend trip ideas')).toBeVisible()
  await expectPageScreenshot(page, 'mobile-chat-drawer.png')
})

test('models on a phone', async ({ page }) => {
  await mockBackend(page)

  await page.goto('/models')
  await expect(page.getByText('deepseek-r1', { exact: true }).first()).toBeVisible()

  await expectPageScreenshot(page, 'mobile-models.png', { fullPage: true })
})
