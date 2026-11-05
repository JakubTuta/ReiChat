import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ref } from 'vue'

// composables/useCopyFeedback.ts is written for Nuxt's auto-import
// environment — see snackbarStore.spec.ts for the same stubbing pattern.
vi.stubGlobal('ref', ref)

const showSnackbarError = vi.fn()
vi.stubGlobal('useSnackbarStore', () => ({ showSnackbarError }))

const { useCopyFeedback } = await import('~/composables/useCopyFeedback')

const writeText = vi.fn<(text: string) => Promise<void>>()

beforeEach(() => {
  vi.useFakeTimers()
  writeText.mockReset()
  showSnackbarError.mockReset()
  vi.stubGlobal('navigator', { clipboard: { writeText } })
})

afterEach(() => {
  vi.useRealTimers()
})

describe('useCopyFeedback', () => {
  it('writes the text and marks that key as copied', async () => {
    writeText.mockResolvedValue()
    const { copiedKey, copy } = useCopyFeedback()

    await copy('docker ps', 'cmd-1')

    expect(writeText).toHaveBeenCalledWith('docker ps')
    expect(copiedKey.value).toBe('cmd-1')
  })

  it('clears the marker after the timeout', async () => {
    writeText.mockResolvedValue()
    const { copiedKey, copy } = useCopyFeedback(1500)

    await copy('x')
    vi.advanceTimersByTime(1499)
    expect(copiedKey.value).toBe('default')

    vi.advanceTimersByTime(1)
    expect(copiedKey.value).toBeNull()
  })

  it('restarts the timeout when something else is copied meanwhile', async () => {
    writeText.mockResolvedValue()
    const { copiedKey, copy } = useCopyFeedback(1000)

    await copy('a', 'first')
    vi.advanceTimersByTime(800)
    await copy('b', 'second')
    vi.advanceTimersByTime(800)

    expect(copiedKey.value).toBe('second')
  })

  it('does not claim success when the browser blocks the clipboard, and says why', async () => {
    writeText.mockRejectedValue(new Error('NotAllowedError'))
    vi.spyOn(console, 'error').mockImplementation(() => {})
    const { copiedKey, copy } = useCopyFeedback()

    await copy('secret')

    expect(copiedKey.value).toBeNull()
    expect(showSnackbarError).toHaveBeenCalledWith(expect.stringContaining('copy it by hand'))
  })
})
