export interface IAIModelVersion {
  parameters: string
  size: string
  size_bytes: number | null
  verified_at: string | null
}

export interface IAIModel {
  id: string
  name: string
  model: string
  description: string
  popularity: number
  can_process_image: boolean
  is_embedding: boolean
  can_use_tools: boolean
  // Added by the user by name rather than listed on ollama.com.
  is_custom: boolean
  versions: IAIModelVersion[]
  index: number
}

export function mapAIModel(data: Partial<IAIModel>): IAIModel {
  return {
    id: data.id?.toString() || '',
    name: data.name || '',
    model: data.model || '',
    description: data.description || '',
    popularity: data.popularity || 0,
    can_process_image: data.can_process_image || false,
    is_embedding: data.is_embedding || false,
    can_use_tools: data.can_use_tools || false,
    is_custom: data.is_custom || false,
    versions: data.versions || [],
    index: data.index || 0,
  }
}
