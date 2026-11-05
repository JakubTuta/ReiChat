// Frozen backend state for the browser tests. Shapes mirror real responses
// from the FastAPI backend, trimmed to what the screens render — the point
// is that a screenshot only changes when the UI changes, never because the
// developer's own models, chats or hardware did.

const GIB = 1024 ** 3

export const hardware = {
  ram_bytes: 16 * GIB,
  vram_bytes: 8 * GIB,
  gpu_available: true,
}

function container(model: string, parameters: string, status: string, port: string | null) {
  return {
    name: `${model}_${parameters}`,
    status,
    port,
    environment: { model, parameters },
  }
}

// One of every card state the Models page can show.
export const containers = [
  container('deepseek-r1', '1.5b', 'running', '11434'),
  container('llama3.1', '8b', 'exited', null),
  container('llava', '7b', 'pulling_model', '11436'),
  container('nomic-embed-text', 'latest', 'incomplete', null),
  container('qwen2.5-coder', '7b', 'restarting', '11437'),
]

export const pullProgress = {
  llava_7b: {
    phase: 'model_pull_progress',
    percent: 42,
    detail: 'pulling 4a5b8f3c1d2e · 2.0 GB / 4.7 GB',
  },
}

export const loadedModels = [
  {
    container: 'deepseek-r1_1.5b',
    model: 'deepseek-r1:1.5b',
    size_bytes: 1_717_000_000,
    vram_bytes: 1_717_000_000,
    context_length: 8192,
    expires_at: null,
  },
]

export const diskUsage = {
  total_bytes: 6_037_000_000,
  models: [
    { model: 'deepseek-r1', parameters: '1.5b', status: 'running', size_bytes: 1_117_322_768 },
    { model: 'llama3.1', parameters: '8b', status: 'exited', size_bytes: 4_920_753_328 },
  ],
}

function model(
  id: number,
  name: string,
  description: string,
  popularity: number,
  versions: [string, string, number][],
  flags: { vision?: boolean, embedding?: boolean, tools?: boolean, custom?: boolean } = {},
) {
  return {
    id,
    name,
    model: name,
    description,
    popularity,
    can_process_image: flags.vision ?? false,
    is_embedding: flags.embedding ?? false,
    can_use_tools: flags.tools ?? false,
    is_custom: flags.custom ?? false,
    versions: versions.map(([parameters, size, size_bytes]) => ({
      parameters,
      size,
      size_bytes,
      verified_at: null,
    })),
    index: id,
  }
}

export const aiModels = [
  model(1, 'llama3.1', 'Llama 3.1 is a state-of-the-art model from Meta available in 8B, 70B and 405B parameter sizes.', 118_300_000, [
    ['8b', '4.6 GB', 4_920_753_328],
    ['70b', '39.6 GB', 42_520_412_561],
    ['405b', '226.4 GB', 243_069_625_106],
  ], { tools: true }),
  model(2, 'deepseek-r1', 'DeepSeek-R1 is a family of open reasoning models with performance approaching that of leading models.', 91_100_000, [
    ['1.5b', '1.0 GB', 1_117_322_768],
    ['7b', '4.4 GB', 4_683_075_440],
    ['14b', '8.4 GB', 8_988_112_209],
    ['32b', '18.5 GB', 19_851_337_809],
  ], { tools: true }),
  model(3, 'llava', 'LLaVA is a multimodal model that combines a vision encoder and Vicuna for general-purpose visual and language understanding.', 8_400_000, [
    ['7b', '4.7 GB', 4_733_351_132],
    ['13b', '8.0 GB', 8_041_000_000],
  ], { vision: true }),
  model(4, 'nomic-embed-text', 'A high-performing open embedding model with a large token context window.', 41_000_000, [
    ['latest', '274 MB', 274_302_450],
  ], { embedding: true }),
  model(5, 'qwen2.5-coder', 'Code-specific Qwen models, a family built for code generation, reasoning and fixing.', 6_200_000, [
    ['7b', '4.7 GB', 4_683_087_332],
    ['32b', '20 GB', 19_851_000_000],
  ], { tools: true, custom: true }),
]

export const chatSummary = {
  id: 1,
  title: 'Quicksort in Python',
  persona: null,
  temperature: null,
  num_ctx: null,
  top_p: null,
  seed: null,
  active_collections: [],
  tools_enabled: false,
  memory_mode: 'trim',
  think_level: '',
  continued_from: null,
}

export const allChats = [
  chatSummary,
  { ...chatSummary, id: 2, title: 'Weekend trip ideas' },
]

export const chatHistory = [
  {
    role: 'user',
    content: 'Write a Python quicksort function',
    image: '',
    thinking: '',
    model: '',
    sibling_count: 1,
    sibling_index: 0,
  },
  {
    role: 'assistant',
    content: [
      'Here is a compact quicksort that picks the middle element as the pivot:',
      '',
      '```python',
      'def quicksort(items):',
      '    if len(items) <= 1:',
      '        return items',
      '    pivot = items[len(items) // 2]',
      '    left = [x for x in items if x < pivot]',
      '    middle = [x for x in items if x == pivot]',
      '    right = [x for x in items if x > pivot]',
      '    return quicksort(left) + middle + quicksort(right)',
      '```',
      '',
      'It runs in **O(n log n)** on average. Things to know:',
      '',
      '- the worst case is O(n²) on already sorted input with a bad pivot',
      '- it is not in-place, so it uses extra memory',
    ].join('\n'),
    image: '',
    thinking: 'The user wants quicksort. A short, readable version is best.',
    model: 'deepseek-r1:1.5b',
    sibling_count: 1,
    sibling_index: 0,
  },
]

export const chatContext = {
  limit: 8192,
  used: 649,
  model_max: 131_072,
  preset: 'standard',
  dropped: 0,
  summarized: false,
  presets: { short: 4096, standard: 8192, long: 16_384, very_long: 32_768 },
  tokens_per_page: 650,
  thinking: { values: [false, true], default: true },
  capabilities: ['tools', 'thinking', 'completion'],
}

export const idleScrape = {
  running: false,
  total: null,
  completed: 0,
  current: null,
  error: null,
}
