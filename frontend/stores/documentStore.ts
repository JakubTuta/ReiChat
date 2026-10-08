import type { IDocument, IDocumentCollection } from '~/models/document'

export const useDocumentStore = defineStore('document', () => {
  const collections = ref<IDocumentCollection[]>([])
  const documentsByCollection = ref<{ [collectionId: number]: IDocument[] }>({})

  const apiStore = useApiStore()
  const snackbarStore = useSnackbarStore()
  const { api } = storeToRefs(apiStore)

  const fetchCollections = async (chatId?: string) => {
    const url = chatId
      ? `collections/?chat_id=${chatId}`
      : 'collections/'

    try {
      const response = await api.value.get(url)

      if (response?.status === 200)
        collections.value = response.data
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || 'Failed to load document collections.')
    }
  }

  const createCollection = async (
    name: string,
    embeddingModelId: number,
    embeddingParameters: string,
    chatId?: string,
  ): Promise<IDocumentCollection | null> => {
    try {
      const response = await api.value.post('collections/', {
        name,
        embedding_model: embeddingModelId,
        embedding_parameters: embeddingParameters,
        ...(chatId
          ? { chat: chatId }
          : {}),
      })

      if (response?.status === 201) {
        collections.value = [response.data, ...collections.value]

        return response.data
      }
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || 'Failed to create collection.')
    }

    return null
  }

  const deleteCollection = async (id: number): Promise<boolean> => {
    try {
      const response = await api.value.delete('collections/', { data: { id } })

      if (response.status === 200) {
        collections.value = collections.value.filter(c => c.id !== id)
        delete documentsByCollection.value[id]

        return true
      }
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || 'Failed to delete collection.')
    }

    return false
  }

  const fetchDocuments = async (collectionId: number) => {
    try {
      const response = await api.value.get('documents/', { params: { collection_id: collectionId } })

      if (response?.status === 200)
        documentsByCollection.value[collectionId] = response.data
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || 'Failed to load documents.')
    }
  }

  const uploadDocument = async (collectionId: number, file: File): Promise<boolean> => {
    try {
      const formData = new FormData()
      formData.append('collection_id', String(collectionId))
      formData.append('file', file)

      // No explicit Content-Type — axios sets multipart/form-data with the
      // right boundary for a FormData body automatically; setting it by
      // hand here would strip that boundary and break parsing server-side.
      const response = await api.value.post('documents/', formData)

      if (response?.status === 202) {
        if (!documentsByCollection.value[collectionId])
          documentsByCollection.value[collectionId] = []

        documentsByCollection.value[collectionId] = [
          response.data,
          ...documentsByCollection.value[collectionId],
        ]
        collections.value = collections.value.map(c => (c.id === collectionId
          ? { ...c, document_count: c.document_count + 1 }
          : c),
        )

        return true
      }
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || 'Failed to upload document.')
    }

    return false
  }

  const fetchChunk = async (
    documentId: number,
    chunkIndex: number,
  ): Promise<{ filename: string, page: number | null, content: string } | { error: string }> => {
    try {
      const response = await api.value.get('documents/chunk/', {
        params: { document_id: documentId, chunk_index: chunkIndex },
      })

      return response.data
    }
    catch (error: any) {
      console.error(error)

      return { error: error.response?.data?.error || 'Could not load that passage.' }
    }
  }

  const deleteDocument = async (collectionId: number, documentId: number): Promise<boolean> => {
    try {
      const response = await api.value.delete('documents/', { data: { id: documentId } })

      if (response.status === 200) {
        documentsByCollection.value[collectionId] = (documentsByCollection.value[collectionId] || [])
          .filter(d => d.id !== documentId)
        collections.value = collections.value.map(c => (c.id === collectionId
          ? { ...c, document_count: Math.max(0, c.document_count - 1) }
          : c),
        )

        return true
      }
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || 'Failed to delete document.')
    }

    return false
  }

  return {
    collections,
    documentsByCollection,
    fetchCollections,
    createCollection,
    deleteCollection,
    fetchDocuments,
    uploadDocument,
    deleteDocument,
    fetchChunk,
  }
})
