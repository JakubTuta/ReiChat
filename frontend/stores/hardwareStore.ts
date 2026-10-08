import type { IDiskUsage, IHardwareInfo } from '~/models/hardware'

export const useHardwareStore = defineStore('hardware', () => {
  const apiStore = useApiStore()
  const snackbarStore = useSnackbarStore()
  const { api } = storeToRefs(apiStore)

  const hardware = ref<IHardwareInfo>({ ram_bytes: null, vram_bytes: null, gpu_available: null })
  const diskUsage = ref<IDiskUsage>({ total_bytes: 0, models: [] })
  const loading = ref(false)

  const fetchHardware = async () => {
    try {
      const response = await api.value.get('docker/hardware/')

      if (response?.status === 200)
        hardware.value = response.data
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || 'Failed to read host hardware info.')
    }
  }

  const fetchDiskUsage = async () => {
    loading.value = true

    try {
      const response = await api.value.get('docker/disk-usage/')

      if (response?.status === 200)
        diskUsage.value = response.data
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || 'Failed to load disk usage.')
    }
    finally {
      loading.value = false
    }
  }

  return {
    hardware,
    diskUsage,
    loading,
    fetchHardware,
    fetchDiskUsage,
  }
})
