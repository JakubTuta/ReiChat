import type { IMCPServer, MCPTestResult, MCPTransport } from '~/models/mcpServer'

export const useMCPServerStore = defineStore('mcpServer', () => {
  const servers = ref<IMCPServer[]>([])
  const loading = ref(false)

  const apiStore = useApiStore()
  const snackbarStore = useSnackbarStore()
  const { api } = storeToRefs(apiStore)

  const fetchServers = async () => {
    loading.value = true

    try {
      const response = await api.value.get('mcp-servers/')

      if (response?.status === 200)
        servers.value = response.data
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || 'Failed to load MCP servers.')
    }
    finally {
      loading.value = false
    }
  }

  const testResults = ref<Record<number, MCPTestResult>>({})
  const testingId = ref<number | null>(null)

  const testServer = async (id: number): Promise<MCPTestResult> => {
    testingId.value = id
    let result: MCPTestResult

    try {
      const response = await api.value.post('mcp-servers/test/', { id })
      result = { ok: true, tools: response.data.tools }
    }
    catch (error: any) {
      console.error(error)
      result = { ok: false, error: error.response?.data?.error || 'Could not test the MCP server.' }
    }
    finally {
      testingId.value = null
    }

    testResults.value = { ...testResults.value, [id]: result }

    return result
  }

  const createServer = async (
    name: string,
    transport: MCPTransport,
    commandOrUrl: string,
  ): Promise<IMCPServer | null> => {
    try {
      const response = await api.value.post('mcp-servers/', {
        name,
        transport,
        ...(transport === 'stdio'
          ? { command: commandOrUrl }
          : { url: commandOrUrl }),
      })

      if (response?.status === 201) {
        servers.value = [...servers.value, response.data].sort((a, b) => a.name.localeCompare(b.name))

        return response.data
      }
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || 'Failed to create MCP server.')
    }

    return null
  }

  const setServerEnabled = async (id: number, enabled: boolean): Promise<boolean> => {
    try {
      const response = await api.value.put('mcp-servers/', { id, enabled })

      if (response?.status === 200) {
        const found = servers.value.find(s => s.id === id)
        if (found)
          found.enabled = enabled

        return true
      }
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || 'Failed to update the MCP server.')
    }

    return false
  }

  const setServerAutoApprove = async (id: number, autoApprove: boolean): Promise<boolean> => {
    try {
      const response = await api.value.put('mcp-servers/', { id, auto_approve: autoApprove })

      if (response?.status === 200) {
        const found = servers.value.find(s => s.id === id)
        if (found)
          found.auto_approve = autoApprove

        return true
      }
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || 'Failed to update the MCP server.')
    }

    return false
  }

  const deleteServer = async (id: number): Promise<boolean> => {
    try {
      const response = await api.value.delete('mcp-servers/', { data: { id } })

      if (response.status === 200) {
        servers.value = servers.value.filter(s => s.id !== id)

        return true
      }
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || 'Failed to delete the MCP server.')
    }

    return false
  }

  return {
    servers,
    loading,
    testResults,
    testingId,
    testServer,
    fetchServers,
    createServer,
    setServerEnabled,
    setServerAutoApprove,
    deleteServer,
  }
})
