import type { VueWrapper } from '@vue/test-utils'
import type { IContainer } from '~/models/container'
import { afterEach, describe, expect, it } from 'vitest'
import ChatStatusBanners from '~/components/ChatStatusBanners.vue'
import { mountApp } from './support/mountApp'

function container(status: string): IContainer {
  return { name: 'Llama', value: 'llama3.1 - 8b', model: 'llama3.1', status, parameters: '8b', canProcessImages: false, canUseTools: false }
}

let wrapper: VueWrapper | null = null

async function mountBanners(props: { model: IContainer | null, reconnecting?: boolean, reconnectFailed?: boolean }) {
  const mounted = await mountApp(ChatStatusBanners, { props: { reconnecting: false, reconnectFailed: false, ...props } })
  wrapper = mounted as unknown as VueWrapper

  return mounted
}

const banners = () => [...document.body.querySelectorAll('.status-banner')].map(b => b.textContent?.replace(/\s+/g, ' ').trim() ?? '')
const manageLink = () => [...document.body.querySelectorAll('a')].find(a => a.textContent?.includes('Manage on Models')) ?? null

afterEach(() => {
  wrapper?.unmount()
  wrapper = null
  document.body.innerHTML = ''
})

describe('chatStatusBanners model state', () => {
  it('stays out of the way when the model is running', async () => {
    await mountBanners({ model: container('running') })

    expect(banners()).toEqual([])
  })

  it('stays out of the way when there is no model at all (the empty state explains that)', async () => {
    await mountBanners({ model: null })

    expect(banners()).toEqual([])
  })

  it.each([
    ['pulling_model', 'still being pulled'],
    ['starting', 'starting up'],
    ['incomplete', 'download didn\'t finish'],
    ['exited', 'not running'],
    ['paused', 'not running'],
  ])('explains a %s model in plain language ("%s")', async (status, phrase) => {
    await mountBanners({ model: container(status) })

    expect(banners()).toHaveLength(1)
    expect(banners()[0]).toContain(phrase)
  })

  it('links to the Models page to fix it, except while a download is already under way', async () => {
    await mountBanners({ model: container('exited') })
    expect(manageLink()?.getAttribute('href')).toBe('/models')
    wrapper!.unmount()
    document.body.innerHTML = ''

    await mountBanners({ model: container('pulling_model') })
    expect(manageLink()).toBeNull()
  })
})

describe('chatStatusBanners connection state', () => {
  it('says the connection dropped while reconnecting, with no button to press yet', async () => {
    await mountBanners({ model: container('running'), reconnecting: true })

    expect(banners()).toHaveLength(1)
    expect(banners()[0]).toContain('Connection lost — reconnecting…')
    expect(document.body.querySelector('button')).toBeNull()
  })

  it('offers a Reconnect button once automatic retries have given up', async () => {
    const mounted = await mountBanners({ model: container('running'), reconnectFailed: true })

    expect(banners()[0]).toContain('Sending is paused')
    document.body.querySelector('button')!.click()

    expect(mounted.emitted('reconnect')).toHaveLength(1)
  })

  it('can show a stopped model and a lost connection together', async () => {
    await mountBanners({ model: container('exited'), reconnecting: true })

    expect(banners()).toHaveLength(2)
  })
})
