import type { ContainerSocketMessage, ContainerSocketWrapper, PullProgressInfo, ScrapeProgressInfo } from '~/constants/containerSocket'
import { getContainerSocket } from '~/constants/containerSocket'

export type ContainerStatus = 'running' | 'exited' | 'paused' | 'restarting' | 'pulling_model' | 'starting' | 'incomplete'

export interface Container {
  name: string
  status: ContainerStatus
  port: string | null
  environment: any
}

export const useContainerStore = defineStore('container', () => {
  const apiStore = useApiStore()
  const snackbarStore = useSnackbarStore()
  const { api } = storeToRefs(apiStore)

  const containers = ref<Container[]>([])
  const loadingOperation = ref<string | null>(null)
  const pullProgress = ref<Record<string, PullProgressInfo>>({})
  const scrapeProgress = ref<ScrapeProgressInfo>({
    running: false,
    total: null,
    completed: 0,
    current: null,
    error: null,
  })

  let pollTimer: ReturnType<typeof setTimeout> | null = null
  let socket: ContainerSocketWrapper | null = null
  let socketConnected = false
  const POLL_INTERVAL_MS = 3000

  const resetState = () => {
    containers.value = []
    loadingOperation.value = null
    pullProgress.value = {}
    stopPolling()
  }

  const shouldPoll = () => !socketConnected
    && containers.value.some(c => ['pulling_model', 'restarting', 'starting'].includes(c.status))

  function stopPolling() {
    if (pollTimer !== null) {
      clearTimeout(pollTimer)
      pollTimer = null
    }
  }

  function scheduleNextPoll() {
    stopPolling()
    if (shouldPoll()) {
      pollTimer = setTimeout(async () => {
        await getUserContainers()
        scheduleNextPoll()
      }, POLL_INTERVAL_MS)
    }
  }

  function startPolling() {
    if (shouldPoll() && pollTimer === null)
      scheduleNextPoll()
  }

  function connectSocket() {
    if (socket)
      return

    socket = getContainerSocket({
      onConnect: () => {
        socketConnected = true
        stopPolling()
      },
      onDisconnect: () => {
        socketConnected = false
      },
      onReconnecting: (_attempt: number) => {
        socketConnected = false
      },
      onMaxReconnectFailed: () => {
        socketConnected = false
        startPolling()
      },
      onMessage: (data: ContainerSocketMessage) => {
        switch (data.type) {
          case 'snapshot':
            if (data.containers)
              containers.value = data.containers as Container[]
            if (data.progress)
              pullProgress.value = data.progress
            if (data.scrape)
              scrapeProgress.value = data.scrape
            break

          case 'image_pull_progress':
          case 'model_pull_progress':
            if (data.container) {
              pullProgress.value = {
                ...pullProgress.value,
                [data.container]: {
                  phase: data.phase || data.type,
                  percent: data.percent ?? 0,
                  detail: data.detail || '',
                },
              }
            }
            break

          case 'scrape_progress':
            scrapeProgress.value = {
              running: data.running ?? false,
              total: data.total ?? null,
              completed: data.completed ?? 0,
              current: data.current ?? null,
              error: data.error ?? null,
            }
            if (!data.running && data.running !== undefined) {
              useChatStore().fetchAIModels()
            }
            break

          case 'container_update':
            if (data.error && data.container) {
              snackbarStore.showSnackbarError(`Container error: ${data.error}`)
            }
            // Snapshot thread will push updated container list shortly; no manual patch needed
            break
        }
      },
    })
  }

  function disconnectSocket() {
    if (socket) {
      socket.closeConnection()
      socket = null
      socketConnected = false
    }
  }

  type SystemStatus = 'ok' | 'docker_down' | 'backend_unreachable'

  const checkSystemStatus = async (): Promise<SystemStatus> => {
    const url = '/docker/'

    try {
      const response = await api.value.get(url)

      return response.status === 200
        ? 'ok'
        : 'docker_down'
    }
    catch (error: any) {
      console.error(error)

      // No `response` at all (or ERR_NETWORK) means the request never got
      // a reply from Django — the backend itself is down, not Docker.
      // A `response` (4xx/5xx) means Django answered and Docker is what's
      // unreachable. Collapsing both into "Docker is not running" sent
      // users to restart Docker Desktop forever when the real problem was
      // the backend container being down.
      if (!error.response || error.code === 'ERR_NETWORK')
        return 'backend_unreachable'

      return 'docker_down'
    }
  }

  // A hoisted function declaration, like scheduleNextPoll/stopPolling
  // above — the two call each other (this reschedules the next poll,
  // scheduleNextPoll's timeout calls this), so neither can be a `const`
  // without one of them referencing the other before it's defined.
  async function getUserContainers() {
    const url = '/docker/containers/'

    try {
      const response = await api.value.get(url)

      if (response.status === 200) {
        containers.value = response.data
        scheduleNextPoll()
      }
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || 'Failed to load containers.')
    }
  }

  const runContainer = async (aiModel: { model: string, parameters: string }) => {
    const url = `/docker/container/${aiModel.model}?parameters=${aiModel.parameters}`
    const operationKey = `run_${aiModel.model}_${aiModel.parameters}`

    loadingOperation.value = operationKey

    try {
      const response = await api.value.post(url, {})

      // 200: model confirmed running. 202: setup/resume started in the
      // background — progress comes via the socket where one is open, and
      // the refresh below starts polling (pulling/starting) where it isn't.
      if (response.status === 200 || response.status === 202) {
        await getUserContainers()
      }
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || `Failed to start container for ${aiModel.model}.`)
    }
    finally {
      if (loadingOperation.value === operationKey)
        loadingOperation.value = null
    }
  }

  const stopContainer = async (aiModel: { model: string, parameters: string }) => {
    const url = `/docker/container/${aiModel.model}?parameters=${aiModel.parameters}&method=stop`
    const operationKey = `stop_${aiModel.model}_${aiModel.parameters}`

    loadingOperation.value = operationKey

    try {
      const response = await api.value.delete(url, {})

      if (response.status === 200) {
        await getUserContainers()
      }
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || `Failed to stop container for ${aiModel.model}.`)
    }
    finally {
      if (loadingOperation.value === operationKey)
        loadingOperation.value = null
    }
  }

  const removeContainer = async (aiModel: { model: string, parameters: string }) => {
    const url = `/docker/container/${aiModel.model}?parameters=${aiModel.parameters}&method=remove`
    const operationKey = `remove_${aiModel.model}_${aiModel.parameters}`

    loadingOperation.value = operationKey

    try {
      const response = await api.value.delete(url, {})

      if (response.status === 200) {
        await getUserContainers()
      }
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || `Failed to remove container for ${aiModel.model}.`)
    }
    finally {
      if (loadingOperation.value === operationKey)
        loadingOperation.value = null
    }
  }

  return {
    containers,
    loadingOperation,
    pullProgress,
    scrapeProgress,
    resetState,
    checkSystemStatus,
    getUserContainers,
    runContainer,
    stopContainer,
    removeContainer,
    startPolling,
    stopPolling,
    connectSocket,
    disconnectSocket,
  }
})
