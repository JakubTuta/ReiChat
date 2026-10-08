export interface IDocumentCollection {
  id: number
  name: string
  chat: number | null
  embedding_model: number
  embedding_model_name: string
  embedding_parameters: string
  document_count: number
  created_at: string
}

export type IDocumentStatus = 'pending' | 'processing' | 'ready' | 'failed'

export interface IDocument {
  id: number
  collection: number
  filename: string
  status: IDocumentStatus
  error_message: string
  chunk_count: number
  // Percent embedded so far, while processing.
  progress: number
  uploaded_at: string
}
