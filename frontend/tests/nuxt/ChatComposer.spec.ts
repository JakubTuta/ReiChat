import type { VueWrapper } from '@vue/test-utils'
import { afterEach, describe, expect, it } from 'vitest'
import ChatComposer from '~/components/ChatComposer.vue'
import { mountApp } from './support/mountApp'

const READY = {
  modelRunning: true,
  waitingForResponse: false,
  sendGateOk: true,
  showImageButton: false,
  message: '',
}

let wrapper: VueWrapper | null = null

// v-model:message — keep the prop in step with what the component emits.
async function mountComposer(props: Partial<typeof READY> = {}) {
  const mounted = await mountApp(ChatComposer, {
    props: {
      ...READY,
      ...props,
      'onUpdate:message': (value: string) => mounted.setProps({ message: value }),
    },
  })
  wrapper = mounted as unknown as VueWrapper

  return mounted
}

const textarea = () => document.body.querySelector<HTMLTextAreaElement>('textarea')!
const button = (label: string) => document.body.querySelector<HTMLButtonElement>(`button[aria-label="${label}"]`)

function pressEscape(target: EventTarget = document.body) {
  target.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }))
}

afterEach(() => {
  wrapper?.unmount()
  wrapper = null
  document.body.innerHTML = ''
})

describe('chatComposer sending', () => {
  it('cannot send an empty message', async () => {
    await mountComposer()

    expect(button('Send message')!.disabled).toBe(true)
  })

  it('cannot send whitespace', async () => {
    await mountComposer({ message: '   \n ' })

    expect(button('Send message')!.disabled).toBe(true)
  })

  it('sends the typed message with the send button', async () => {
    const mounted = await mountComposer({ message: 'hello there' })

    button('Send message')!.click()

    expect(mounted.emitted('send')).toEqual([[{ message: 'hello there', image: '' }]])
  })

  it('sends on Enter but keeps Shift+Enter for a new line', async () => {
    const mounted = await mountComposer({ message: 'line one' })

    textarea().dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', shiftKey: true, bubbles: true, cancelable: true }))
    expect(mounted.emitted('send')).toBeUndefined()

    textarea().dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }))
    expect(mounted.emitted('send')).toHaveLength(1)
  })

  it('does not send while the connection is down, even with text typed', async () => {
    await mountComposer({ message: 'hi', sendGateOk: false })

    expect(button('Send message')!.disabled).toBe(true)
  })

  it('is disabled with a stopped model instead of silently swallowing input', async () => {
    await mountComposer({ modelRunning: false, message: 'hi' })

    expect(textarea().disabled).toBe(true)
    expect(button('Send message')!.disabled).toBe(true)
  })

  it('only offers image attachments for models that can see them', async () => {
    await mountComposer({ showImageButton: false })
    expect(button('Attach image')).toBeNull()
    wrapper!.unmount()
    document.body.innerHTML = ''

    await mountComposer({ showImageButton: true })
    expect(button('Attach image')).not.toBeNull()
  })
})

describe('chatComposer while a reply is streaming', () => {
  it('swaps send for a stop button that stops', async () => {
    const mounted = await mountComposer({ waitingForResponse: true })

    expect(button('Send message')).toBeNull()
    button('Stop generating')!.click()

    expect(mounted.emitted('stop')).toHaveLength(1)
  })

  it('stops when Escape is pressed, as the "esc to stop" hint promises', async () => {
    const mounted = await mountComposer({ waitingForResponse: true })

    expect(document.body.textContent).toContain('esc to stop')
    pressEscape()

    expect(mounted.emitted('stop')).toHaveLength(1)
  })

  it('ignores Escape when nothing is streaming', async () => {
    const mounted = await mountComposer({ waitingForResponse: false })

    pressEscape()

    expect(mounted.emitted('stop')).toBeUndefined()
  })

  it('leaves Escape to another text field, such as renaming a chat', async () => {
    const mounted = await mountComposer({ waitingForResponse: true })
    const renameField = document.createElement('input')
    document.body.append(renameField)

    pressEscape(renameField)

    expect(mounted.emitted('stop')).toBeUndefined()
  })

  it('leaves Escape to an open dialog, which should close first', async () => {
    const mounted = await mountComposer({ waitingForResponse: true })
    const openDialog = document.createElement('div')
    openDialog.className = 'v-overlay v-dialog v-overlay--active'
    document.body.append(openDialog)

    pressEscape()

    expect(mounted.emitted('stop')).toBeUndefined()
  })

  it('stops listening once the composer is gone', async () => {
    const mounted = await mountComposer({ waitingForResponse: true })
    mounted.unmount()

    pressEscape()

    expect(mounted.emitted('stop')).toBeUndefined()
  })
})
