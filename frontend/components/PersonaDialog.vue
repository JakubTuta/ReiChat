<script setup lang="ts">
import type { IPersona } from '~/models/persona'

const props = defineProps<{
  currentPersonaId: string | null
}>()

const emit = defineEmits<{
  (e: 'select', persona: { id: string, name: string } | null): void
}>()

const isOpen = defineModel<boolean>({ default: false, required: true })

const personaStore = usePersonaStore()
const { personas, loading } = storeToRefs(personaStore)

const mode = ref<'list' | 'edit'>('list')
const editingId = ref<string | null>(null)
const formName = ref('')
const formPrompt = ref('')
const saving = ref(false)

const deleteConfirmDialog = ref(false)
const personaToDelete = ref<IPersona | null>(null)

watch(isOpen, (open) => {
  if (open && !personas.value.length)
    personaStore.fetchPersonas()

  if (!open) {
    mode.value = 'list'
    editingId.value = null
  }
})

function openCreate() {
  mode.value = 'edit'
  editingId.value = null
  formName.value = ''
  formPrompt.value = ''
}

function openEdit(persona: IPersona) {
  mode.value = 'edit'
  editingId.value = persona.id
  formName.value = persona.name
  formPrompt.value = persona.system_prompt
}

function cancelEdit() {
  mode.value = 'list'
  editingId.value = null
}

async function saveForm() {
  const name = formName.value.trim()
  const systemPrompt = formPrompt.value.trim()
  if (!name || !systemPrompt)
    return

  saving.value = true

  if (editingId.value)
    await personaStore.updatePersona(editingId.value, name, systemPrompt)
  else
    await personaStore.createPersona(name, systemPrompt)

  saving.value = false
  cancelEdit()
}

function requestDelete(persona: IPersona) {
  personaToDelete.value = persona
  deleteConfirmDialog.value = true
}

async function confirmDelete() {
  deleteConfirmDialog.value = false

  if (!personaToDelete.value)
    return

  const deletedId = personaToDelete.value.id
  const deleted = await personaStore.deletePersona(deletedId)
  personaToDelete.value = null

  // The chat pointing at a persona that no longer exists needs to fall back
  // to "no persona" locally too — the backend already did this via SET_NULL.
  if (deleted && props.currentPersonaId === deletedId)
    emit('select', null)
}

function cancelDelete() {
  deleteConfirmDialog.value = false
  personaToDelete.value = null
}

function selectPersona(persona: IPersona | null) {
  emit('select', persona
    ? { id: persona.id, name: persona.name }
    : null)
  isOpen.value = false
}
</script>

<template>
  <v-dialog
    v-model="isOpen"
    max-width="480"
    transition="dialog-rise-transition"
  >
    <v-card class="reichat-dialog">
      <div class="reichat-dialog-header">
        <span class="reichat-dialog-title">
          {{ mode === 'list'
            ? 'Persona'
            : editingId
              ? 'Edit persona'
              : 'New persona' }}
        </span>

        <button
          type="button"
          class="reichat-dialog-close"
          title="Close"
          aria-label="Close"
          @click="isOpen = false"
        >
          <v-icon
            size="16"
            icon="$mdiClose"
          />
        </button>
      </div>

      <v-card-text class="reichat-dialog-body">
        <template v-if="mode === 'list'">
          <div
            class="persona-row"
            :class="{'persona-row--active': !currentPersonaId}"
            @click="selectPersona(null)"
          >
            <span class="persona-row-name">No persona (default)</span>
          </div>

          <div
            v-for="persona in personas"
            :key="persona.id"
            class="persona-row"
            :class="{'persona-row--active': currentPersonaId === persona.id}"
            @click="selectPersona(persona)"
          >
            <div class="persona-row-main">
              <span class="persona-row-name">{{ persona.name }}</span>

              <span class="persona-row-prompt">{{ persona.system_prompt }}</span>
            </div>

            <div class="persona-row-actions">
              <button
                type="button"
                class="reichat-dialog-close"
                title="Edit persona"
                aria-label="Edit persona"
                @click.stop="openEdit(persona)"
              >
                <v-icon
                  size="14"
                  icon="$mdiPencil"
                />
              </button>

              <button
                type="button"
                class="reichat-dialog-close persona-row-delete"
                title="Delete persona"
                aria-label="Delete persona"
                @click.stop="requestDelete(persona)"
              >
                <v-icon
                  size="14"
                  icon="$mdiDelete"
                />
              </button>
            </div>
          </div>

          <div
            v-if="!loading && !personas.length"
            class="persona-empty"
          >
            No saved personas yet — create one below.
          </div>

          <button
            type="button"
            class="dashed-add-row"
            @click="openCreate"
          >
            + New persona
          </button>
        </template>

        <template v-else>
          <v-text-field
            v-model="formName"
            label="Name"
            density="compact"
            autofocus
            class="mb-2"
          />

          <v-textarea
            v-model="formPrompt"
            label="System prompt"
            auto-grow
            rows="3"
            hide-details
          />
        </template>
      </v-card-text>

      <v-card-actions
        v-if="mode === 'edit'"
        class="reichat-dialog-actions"
      >
        <v-btn
          variant="text"
          @click="cancelEdit"
        >
          Cancel
        </v-btn>

        <v-spacer />

        <v-btn
          color="mint-btn"
          variant="flat"
          :loading="saving"
          :disabled="!formName.trim() || !formPrompt.trim()"
          @click="saveForm"
        >
          Save
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>

  <ConfirmDialog
    v-model="deleteConfirmDialog"
    title="Delete persona"
    message="Any chat using this persona falls back to no persona. This can't be undone."
    confirm-label="Delete"
    confirm-color="red"
    @confirm="confirmDelete"
    @cancel="cancelDelete"
  />
</template>

<style scoped>
.persona-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 9px 11px;
  border-radius: 10px;
  cursor: pointer;
  transition: background-color 0.15s ease;
}

.persona-row:hover {
  background: var(--color-soft);
}

.persona-row--active {
  background: var(--color-mint-tint);
}

.persona-row-main {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.persona-row-name {
  font-size: 12.5px;
  font-weight: 500;
  color: var(--color-ink);
}

.persona-row-prompt {
  font-size: 11.5px;
  color: var(--color-ink-2);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.persona-row-actions {
  display: flex;
  gap: 2px;
  flex-shrink: 0;
  opacity: 0;
  transition: opacity 0.15s ease;
}

.persona-row:hover .persona-row-actions,
.persona-row:focus-within .persona-row-actions {
  opacity: 1;
}

.persona-row-delete:hover {
  color: var(--color-red);
}

.persona-empty {
  padding: 16px 4px;
  text-align: center;
  font-size: 12.5px;
  color: var(--color-ink-2);
}
</style>
