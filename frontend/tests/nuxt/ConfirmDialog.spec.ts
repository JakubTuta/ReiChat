import { afterEach, describe, expect, it } from 'vitest'
import ConfirmDialog from '~/components/ConfirmDialog.vue'
import { mountApp } from './support/mountApp'

afterEach(() => {
  document.body.innerHTML = ''
})

// v-dialog renders into an overlay container on <body>, outside the wrapper.
function dialogButton(label: string): HTMLElement {
  const button = [...document.body.querySelectorAll<HTMLElement>('.reichat-dialog button')]
    .find(b => b.textContent?.trim() === label)
  if (!button)
    throw new Error(`No "${label}" button in the dialog`)

  return button
}

function openDialog(props: Record<string, unknown> = {}) {
  return mountApp(ConfirmDialog, {
    props: { modelValue: true, title: 'Delete chat', message: 'This cannot be undone.', ...props },
  })
}

describe('confirmDialog', () => {
  it('shows the title and message so the user knows what they are confirming', async () => {
    await openDialog()

    const dialog = document.body.querySelector('.reichat-dialog')
    expect(dialog?.textContent).toContain('Delete chat')
    expect(dialog?.textContent).toContain('This cannot be undone.')
  })

  it('defaults the confirm button to "Delete", the destructive wording the app uses', async () => {
    await openDialog()

    expect(dialogButton('Delete')).toBeTruthy()
  })

  it('uses a custom confirm label', async () => {
    await openDialog({ confirmLabel: 'Remove' })

    expect(dialogButton('Remove')).toBeTruthy()
  })

  it('emits confirm only when the confirm button is clicked', async () => {
    const wrapper = await openDialog()

    dialogButton('Cancel').click()
    expect(wrapper.emitted('confirm')).toBeUndefined()

    dialogButton('Delete').click()
    expect(wrapper.emitted('confirm')).toHaveLength(1)
  })

  it('emits cancel from the cancel button, without confirming', async () => {
    const wrapper = await openDialog()

    dialogButton('Cancel').click()

    expect(wrapper.emitted('cancel')).toHaveLength(1)
    expect(wrapper.emitted('confirm')).toBeUndefined()
  })

  it('renders nothing while closed', async () => {
    await openDialog({ modelValue: false })

    expect(document.body.querySelector('.reichat-dialog')).toBeNull()
  })
})
