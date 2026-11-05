import type { VueWrapper } from '@vue/test-utils'
import type { IAIModel, IAIModelVersion } from '~/models/aiModel'
import { afterEach, describe, expect, it } from 'vitest'
import ModelCard from '~/components/ModelCard.vue'
import { mapAIModel } from '~/models/aiModel'
import { mountApp } from './support/mountApp'

const GB = 1024 ** 3

const v8b: IAIModelVersion = { parameters: '8b', size: '4.6 GB', size_bytes: 4.6 * GB, verified_at: null }
const v70b: IAIModelVersion = { parameters: '70b', size: '39.6 GB', size_bytes: 39.6 * GB, verified_at: null }

const model: IAIModel = mapAIModel({
  id: '1',
  name: 'llama3.1',
  model: 'llama3.1',
  description: 'A model.',
  popularity: 118_300_000,
  versions: [v8b, v70b],
})

const baseProps = {
  model,
  installedVersionCount: 0,
  status: 'not_found',
  pullProgress: null,
  operationLoading: false,
  loaded: null,
  hardware: { ram_bytes: 16 * GB, vram_bytes: 8 * GB, gpu_available: true },
  isExternal: false,
  manualCommands: [] as string[],
  selectedVersion: v8b as IAIModelVersion | null,
  shell: 'bash',
}

let wrapper: VueWrapper | null = null

async function mountCard(props: Record<string, unknown> = {}) {
  const mounted = await mountApp(ModelCard, { props: { ...baseProps, ...props } })
  wrapper = mounted as unknown as VueWrapper

  return mounted
}

function buttonLabelled(label: string): HTMLButtonElement | null {
  return [...document.body.querySelectorAll<HTMLButtonElement>('button')]
    .find(b => b.textContent?.trim() === label) ?? null
}

function requireButton(label: string): HTMLButtonElement {
  const button = buttonLabelled(label)
  if (!button)
    throw new Error(`No "${label}" button on the card`)

  return button
}

afterEach(() => {
  wrapper?.unmount()
  wrapper = null
  document.body.innerHTML = ''
})

describe('modelCard primary action', () => {
  it.each([
    ['not_found', 'Create container', 'create'],
    ['exited', 'Start container', 'start'],
    ['paused', 'Start container', 'start'],
    ['incomplete', 'Resume download', 'resume'],
    ['running', 'Stop container', 'stop'],
  ])('a %s container offers "%s", which emits %s', async (status, label, event) => {
    const mounted = await mountCard({ status })

    const button = requireButton(label)
    expect(button.disabled).toBe(false)
    button.click()

    expect(mounted.emitted(event)).toHaveLength(1)
  })

  it.each([
    ['starting', 'Starting…'],
    ['pulling_model', 'Pulling model…'],
    ['restarting', 'Please wait…'],
  ])('a %s container is busy: the action is disabled and says "%s"', async (status, label) => {
    const mounted = await mountCard({ status })

    const button = requireButton(label)
    expect(button.disabled).toBe(true)
    button.click()
    expect(mounted.emitted()).not.toHaveProperty('create')
    expect(mounted.emitted()).not.toHaveProperty('stop')
  })

  it('shows "Downloading…" and the progress while a model downloads, and hides update checks', async () => {
    await mountCard({
      status: 'pulling_model',
      pullProgress: { phase: 'model_pull_progress', percent: 42, detail: '2.0 GB / 4.7 GB' },
    })

    expect(requireButton('Downloading…').disabled).toBe(true)
    expect(document.body.textContent).toContain('Downloading model')
    expect(document.body.textContent).toContain('42%')
    expect(document.body.textContent).toContain('2.0 GB / 4.7 GB')
    expect(buttonLabelled('Check for update')).toBeNull()
  })

  it('says "Downloading base image" for the image phase', async () => {
    await mountCard({
      status: 'pulling_model',
      pullProgress: { phase: 'image_pull_progress', percent: 10, detail: 'ollama/ollama' },
    })

    expect(document.body.textContent).toContain('Downloading base image')
  })

  it('cannot start or remove a size while another action on it is in flight', async () => {
    await mountCard({ status: 'running', operationLoading: true })

    expect(requireButton('Please wait…').disabled).toBe(true)
    expect(requireButton('Remove').disabled).toBe(true)
  })

  it('cannot stop a model that lives in the user\'s own Ollama', async () => {
    await mountCard({ status: 'running', isExternal: true })

    expect(requireButton('Installed').disabled).toBe(true)
    expect(buttonLabelled('Stop container')).toBeNull()
  })
})

describe('modelCard secondary actions', () => {
  it('offers an update check only for a running model, and emits update', async () => {
    const mounted = await mountCard({ status: 'running' })

    requireButton('Check for update').click()
    expect(mounted.emitted('update')).toHaveLength(1)
    wrapper!.unmount()
    document.body.innerHTML = ''

    await mountCard({ status: 'exited' })
    expect(buttonLabelled('Check for update')).toBeNull()
  })

  it('offers removal for any container that exists, and emits remove', async () => {
    const mounted = await mountCard({ status: 'exited' })

    requireButton('Remove').click()
    expect(mounted.emitted('remove')).toHaveLength(1)
    wrapper!.unmount()
    document.body.innerHTML = ''

    await mountCard({ status: 'not_found' })
    expect(buttonLabelled('Remove')).toBeNull()
  })

  it('lets the user free the memory a loaded model holds', async () => {
    const mounted = await mountCard({
      status: 'running',
      loaded: { container: 'llama3.1_8b', model: 'llama3.1:8b', size_bytes: 4.6 * GB, vram_bytes: 4.6 * GB, context_length: 8192, expires_at: null },
    })

    expect(document.body.textContent).toContain('In memory')
    expect(document.body.textContent).toContain('fully on GPU')
    requireButton('Free memory').click()

    expect(mounted.emitted('unload')).toHaveLength(1)
  })
})

describe('modelCard sizes', () => {
  it('tells the user to pick a size first, and offers no actions until they do', async () => {
    await mountCard({ selectedVersion: null })

    expect(document.body.textContent).toContain('Select a version to manage this model.')
    expect(buttonLabelled('Create container')).toBeNull()
  })

  it('selects a size on click and deselects it on a second click', async () => {
    const mounted = await mountCard({
      'selectedVersion': null,
      'onUpdate:selectedVersion': (value: IAIModelVersion | null) => mounted.setProps({ selectedVersion: value }),
    })

    requireButton('70b · 39.6 GB').click()
    expect(mounted.emitted('update:selectedVersion')?.slice(-1)[0]).toEqual([v70b])

    await mounted.setProps({ selectedVersion: v70b })
    requireButton('70b · 39.6 GB').click()
    expect(mounted.emitted('update:selectedVersion')?.slice(-1)[0]).toEqual([null])
  })

  it('warns when the selected size will not fit this machine', async () => {
    await mountCard({ selectedVersion: v70b })

    expect(document.body.textContent).toContain('won\'t fit')
  })

  it('stays quiet about fit when the machine\'s memory is unknown', async () => {
    await mountCard({ hardware: { ram_bytes: null, vram_bytes: null, gpu_available: null } })

    expect(document.body.textContent).not.toContain('runs well')
    expect(document.body.textContent).not.toContain('won\'t fit')
  })
})

describe('modelCard header', () => {
  it('shows what the model is and how many sizes are installed', async () => {
    await mountCard({ installedVersionCount: 2 })

    expect(document.body.textContent).toContain('llama3.1')
    expect(document.body.textContent).toContain('2 INSTALLED')
    expect(document.body.textContent).toContain('★ 118.3M')
  })

  it('marks vision models and models the user added themselves', async () => {
    await mountCard({ model: { ...model, can_process_image: true, is_custom: true } })

    expect(document.body.textContent).toContain('VISION')
    expect(document.body.textContent).toContain('ADDED BY YOU')
  })

  it('does not call a running embedding model "ready to chat"', async () => {
    await mountCard({ status: 'running', model: { ...model, is_embedding: true } })

    expect(document.body.textContent).toContain('used for searching your files')
    expect(document.body.textContent).not.toContain('ready to chat')
  })
})

describe('modelCard manual commands', () => {
  it('shows the run-manually panel only with commands and only for managed containers', async () => {
    await mountCard({ manualCommands: ['docker stop llama3.1_8b'] })
    expect(document.body.textContent).toContain('advanced: run manually')
    wrapper!.unmount()
    document.body.innerHTML = ''

    await mountCard({ manualCommands: [] })
    expect(document.body.textContent).not.toContain('advanced: run manually')
    wrapper!.unmount()
    document.body.innerHTML = ''

    await mountCard({ manualCommands: ['docker stop llama3.1_8b'], isExternal: true })
    expect(document.body.textContent).not.toContain('advanced: run manually')
  })
})
