import type { Page } from '@playwright/test'
import type { MockedBackend } from './support/mockBackend'
import { expect, test } from '@playwright/test'
import * as fixtures from './support/fixtures'
import { mockBackend } from './support/mockBackend'
import { expectPageScreenshot } from './support/screenshot'

async function openChat(page: Page, overrides: Parameters<typeof mockBackend>[1] = {}): Promise<MockedBackend> {
  const backend = await mockBackend(page, overrides)

  await page.goto('/chat')
  await expect(page.getByPlaceholder('Message…')).toBeVisible()

  return backend
}

async function pickFromPlusMenu(page: Page, row: string) {
  await page.getByRole('button', { name: 'Add to this chat' }).click()
  await page.locator('.plus-row', { hasText: row }).click()
  await expect(page.getByRole('dialog')).toBeVisible()
}

test('an empty chat offers starter suggestions', async ({ page }) => {
  await openChat(page, { chatHistory: [] })

  await expectPageScreenshot(page, 'chat-empty.png')
})

test('a stopped model explains why sending is disabled', async ({ page }) => {
  const stopped = fixtures.containers.map((c) => {
    return c.name === 'deepseek-r1_1.5b'
      ? { ...c, status: 'exited' }
      : c
  })
  await openChat(page, { containers: stopped })

  await expect(page.getByText('Container is not running.')).toBeVisible()
  await expectPageScreenshot(page, 'chat-model-stopped.png')
})

test('a reply streaming in with thinking and a tool call', async ({ page }) => {
  const backend = await openChat(page)
  await backend.chatConnected

  backend.pushChatFrame({ message: '', done: false, thinking: 'The user wants the time, so I should call the clock tool.' })
  backend.pushChatFrame({
    message: '',
    done: false,
    tool_call: { name: 'get_current_time', args: {}, result: '14:32' },
  })
  backend.pushChatFrame({ message: 'It is currently 14:32', done: false })
  await expect(page.getByText('It is currently 14:32')).toBeVisible()

  await expectPageScreenshot(page, 'chat-streaming.png')
})

test('a tool call waits for the user to approve it', async ({ page }) => {
  const backend = await openChat(page)
  await backend.chatConnected

  backend.pushChatFrame({
    message: '',
    done: false,
    tool_approval: { id: 'call-1', name: 'read_file', server: 'filesystem', args: { path: '/notes/todo.md' } },
  })
  await expect(page.getByRole('button', { name: /Allow/ }).first()).toBeVisible()

  await expectPageScreenshot(page, 'chat-tool-approval.png')
})

test('a dropped connection pauses sending', async ({ page }) => {
  await mockBackend(page)
  await page.routeWebSocket(/\/ws\/chat\/[^/]+\/$/, ws => ws.close())

  await page.goto('/chat')
  await expect(page.getByText('Connection lost — reconnecting…')).toBeVisible()

  await expectPageScreenshot(page, 'chat-reconnecting.png')
})

test('editing an earlier message', async ({ page }) => {
  await openChat(page)

  await page.getByRole('button', { name: 'edit & resend' }).click()
  await expect(page.getByRole('textbox').first()).toBeVisible()

  await expectPageScreenshot(page, 'chat-editing.png')
})

test('deleting a chat asks for confirmation', async ({ page }) => {
  await openChat(page)

  await page.getByRole('button', { name: 'Delete chat' }).first().click()
  await expect(page.getByRole('dialog')).toBeVisible()

  await expectPageScreenshot(page, 'chat-delete-confirm.png')
})

test('search chats dialog', async ({ page }) => {
  await openChat(page)

  await page.getByRole('button', { name: 'Search chats' }).click()
  await expect(page.getByRole('dialog')).toBeVisible()

  await expectPageScreenshot(page, 'chat-search.png')
})

test('the add-to-chat menu', async ({ page }) => {
  await openChat(page)

  await page.getByRole('button', { name: 'Add to this chat' }).click()
  await expect(page.locator('.plus-popover')).toBeVisible()

  await expectPageScreenshot(page, 'chat-plus-menu.png')
})

const dialogs = [
  { name: 'persona', row: 'Persona' },
  { name: 'files', row: 'Files' },
  { name: 'parameters', row: 'Parameters' },
  { name: 'prompt-template', row: 'Prompt template' },
  { name: 'json-output', row: 'JSON output' },
] as const

for (const { name, row } of dialogs) {
  test(`${name} dialog`, async ({ page }) => {
    await openChat(page)

    await pickFromPlusMenu(page, row)

    await expectPageScreenshot(page, `chat-dialog-${name}.png`)
  })
}

test('memory dialog opens from the memory meter', async ({ page }) => {
  await openChat(page)

  await page.locator('.memory-meter').click()
  await expect(page.getByRole('dialog')).toBeVisible()

  await expectPageScreenshot(page, 'chat-dialog-memory.png')
})

test('the drawer collapses from the thread header', async ({ page }) => {
  await openChat(page)

  await page.getByRole('button', { name: 'Toggle conversations list' }).click()

  await expectPageScreenshot(page, 'chat-drawer-closed.png')
})
