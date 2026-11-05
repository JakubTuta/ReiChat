import type { MessagePart } from '~/utils/splitMessage'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import MessageParts from '~/components/MessageParts.vue'
import { mountApp } from './support/mountApp'

// Sanitising hostile model output is NOT tested here: DOMPurify reports itself
// unsupported in the fake DOM these tests run in and returns its input
// untouched, so a test would pass or fail for the wrong reason. It is covered
// in a real browser by tests/browser/security.spec.ts.

function mountParts(parts: MessagePart[]) {
  return mountApp(MessageParts, { props: { parts } })
}

const code: MessagePart[] = [{ title: 'code', language: 'html', content: '<b onclick="x()">not bold</b>' }]

afterEach(() => {
  document.body.innerHTML = ''
})

describe('messageParts markdown', () => {
  it('renders ordinary markdown', async () => {
    await mountParts([{ title: 'text', content: '**bold** and a list:\n\n- one\n- two\n\n[docs](https://ollama.com)' }])

    expect(document.body.querySelector('strong')?.textContent).toBe('bold')
    expect(document.body.querySelectorAll('li')).toHaveLength(2)
    expect(document.body.querySelector('a')?.getAttribute('href')).toBe('https://ollama.com')
  })

  it('keeps single line breaks, as a chat reply expects', async () => {
    await mountParts([{ title: 'text', content: 'first line\nsecond line' }])

    expect(document.body.querySelector('br')).not.toBeNull()
  })

  it('renders each part in order, text and code interleaved', async () => {
    await mountParts([
      { title: 'text', content: 'before' },
      { title: 'code', language: 'python', content: 'print(1)' },
      { title: 'text', content: 'after' },
    ])

    const order = [...document.body.querySelectorAll('.markdown-body, .code-block')].map(el => el.className.split(' ')[0])
    expect(order).toEqual(['markdown-body', 'code-block', 'markdown-body'])
  })
})

describe('messageParts code blocks', () => {
  it('shows code as text, never as markup', async () => {
    await mountParts(code)
    const block = document.body.querySelector('.code-text')!

    expect(block.textContent).toBe('<b onclick="x()">not bold</b>')
    expect(block.querySelector('b')).toBeNull()
  })

  it('labels the language, falling back to "code"', async () => {
    await mountParts(code)
    expect(document.body.querySelector('.code-title')!.textContent).toContain('html')
    document.body.innerHTML = ''

    await mountParts([{ title: 'code', content: 'x' }])
    expect(document.body.querySelector('.code-title')!.textContent).toContain('code')
  })

  describe('copy button', () => {
    const writeText = vi.fn<(text: string) => Promise<void>>()

    beforeEach(() => {
      writeText.mockReset()
      // Only the clipboard: Vuetify reads navigator.userAgent at setup.
      Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true })
    })

    it('copies the raw code and confirms once it worked', async () => {
      writeText.mockResolvedValue()
      await mountParts(code)
      const button = document.body.querySelector<HTMLButtonElement>('.code-copy')!

      button.click()
      await vi.waitFor(() => expect(button.textContent?.trim()).toBe('copied'))

      expect(writeText).toHaveBeenCalledWith('<b onclick="x()">not bold</b>')
    })

    it('does not say "copied" when the browser refuses', async () => {
      writeText.mockRejectedValue(new Error('NotAllowedError'))
      vi.spyOn(console, 'error').mockImplementation(() => {})
      await mountParts(code)
      const button = document.body.querySelector<HTMLButtonElement>('.code-copy')!

      button.click()
      await vi.waitFor(() => expect(writeText).toHaveBeenCalled())

      expect(button.textContent?.trim()).toBe('copy')
    })
  })
})
