import type { Page, WebSocketRoute } from '@playwright/test'
import * as fixtures from './fixtures'

const API_ORIGIN = 'http://localhost:8000'

const CORS_HEADERS = {
  'access-control-allow-origin': '*',
  'access-control-allow-methods': '*',
  'access-control-allow-headers': '*',
}

export interface BackendData {
  containers: unknown[]
  pullProgress: Record<string, unknown>
  loadedModels: unknown[]
  aiModels: unknown[]
  allChats: unknown[]
  chatHistory: unknown[]
  chatContext: unknown
}

export function defaultData(): BackendData {
  return {
    containers: fixtures.containers,
    pullProgress: fixtures.pullProgress,
    loadedModels: fixtures.loadedModels,
    aiModels: fixtures.aiModels,
    allChats: fixtures.allChats,
    chatHistory: fixtures.chatHistory,
    chatContext: fixtures.chatContext,
  }
}

export interface MockedBackend {
  // Backend calls no fixture answered. A browser test that reaches the real
  // backend (or hits a 404) would screenshot the developer's own data, so
  // tests assert this stays empty.
  unmocked: string[]
  // Console reports of the page's Content-Security-Policy blocking something.
  // The app must never trip its own policy; a reply that tries to is expected to.
  cspViolations: string[]
  // Resolves once the page has opened its chat socket.
  chatConnected: Promise<void>
  // Streams a frame to the open chat socket, as the server would mid-reply.
  pushChatFrame: (frame: Record<string, unknown>) => void
}

type Json = Record<string, unknown> | unknown[]

interface Route {
  method: string
  path: RegExp
  body: (data: BackendData) => Json
}

const routes: Route[] = [
  { method: 'GET', path: /^\/docker\/$/, body: () => ({ status: 'Docker is running' }) },
  { method: 'GET', path: /^\/docker\/hardware\/$/, body: () => fixtures.hardware },
  { method: 'GET', path: /^\/docker\/disk-usage\/$/, body: () => fixtures.diskUsage },
  { method: 'GET', path: /^\/docker\/containers\/$/, body: d => d.containers as unknown[] },
  { method: 'GET', path: /^\/docker\/runtime\/$/, body: d => ({ loaded: d.loadedModels }) },
  { method: 'GET', path: /^\/docker\/ollama-connection\/$/, body: () => ({ mode: 'managed', url: null, version: null }) },
  { method: 'GET', path: /^\/docker\/web-search\/$/, body: () => ({ status: 'off' }) },
  { method: 'POST', path: /^\/docker\/container\/[^/]+$/, body: () => ({}) },
  { method: 'GET', path: /^\/ai-models\/$/, body: d => d.aiModels as unknown[] },
  { method: 'GET', path: /^\/all-chats\/[^/]+$/, body: d => d.allChats as unknown[] },
  { method: 'GET', path: /^\/chat-history\/[^/]+\/[^/]+$/, body: d => d.chatHistory as unknown[] },
  { method: 'GET', path: /^\/chat-context\/[^/]+\/[^/]+$/, body: d => d.chatContext as Json },
  { method: 'GET', path: /^\/personas\/$/, body: () => [] },
  { method: 'GET', path: /^\/prompt-templates\/$/, body: () => [] },
  { method: 'GET', path: /^\/mcp-servers\/$/, body: () => [] },
  { method: 'GET', path: /^\/collections\/$/, body: () => [] },
  { method: 'GET', path: /^\/compare-votes\/$/, body: () => [] },
]

export async function mockBackend(page: Page, overrides: Partial<BackendData> = {}): Promise<MockedBackend> {
  const data: BackendData = { ...defaultData(), ...overrides }
  let chatSocket: WebSocketRoute | null = null
  let markChatConnected: () => void = () => {}

  const state: MockedBackend = {
    unmocked: [],
    cspViolations: [],
    chatConnected: new Promise<void>((resolve) => {
      markChatConnected = resolve
    }),
    pushChatFrame: (frame) => {
      if (!chatSocket)
        throw new Error('The chat socket is not open yet — await chatConnected first.')

      chatSocket.send(JSON.stringify(frame))
    },
  }

  page.on('console', (message) => {
    if (message.text().includes('Content Security Policy'))
      state.cspViolations.push(message.text())
  })

  await page.route(`${API_ORIGIN}/**`, async (route) => {
    const request = route.request()
    const { pathname } = new URL(request.url())

    // The app is on another origin than the backend, so JSON writes are
    // preceded by a CORS preflight that has to be answered too.
    if (request.method() === 'OPTIONS') {
      await route.fulfill({ status: 204, headers: CORS_HEADERS })

      return
    }

    const match = routes.find(r => r.method === request.method() && r.path.test(pathname))

    if (!match) {
      state.unmocked.push(`${request.method()} ${pathname}`)
      await route.fulfill({
        status: 404,
        contentType: 'application/json',
        headers: CORS_HEADERS,
        body: JSON.stringify({ error: `No browser-test fixture for ${request.method()} ${pathname}` }),
      })

      return
    }

    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      headers: CORS_HEADERS,
      body: JSON.stringify(match.body(data)),
    })
  })

  await page.routeWebSocket(/\/ws\/containers\/$/, (ws) => {
    ws.send(JSON.stringify({
      type: 'snapshot',
      containers: data.containers,
      progress: data.pullProgress,
      scrape: fixtures.idleScrape,
    }))
  })

  await page.routeWebSocket(/\/ws\/chat\/[^/]+\/$/, (ws) => {
    chatSocket = ws
    markChatConnected()
  })

  // Nothing is generated in a compare test, so the socket only has to open.
  await page.routeWebSocket(/\/ws\/compare\/$/, () => {})

  return state
}
