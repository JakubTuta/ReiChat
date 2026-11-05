<script setup lang="ts">
import type { IPromptTemplate } from '~/models/promptTemplate'
import { extractTemplateVariables, renderTemplate } from '~/utils/promptTemplate'

const emit = defineEmits<{
  (e: 'insert', content: string): void
}>()

const isOpen = defineModel<boolean>({ default: false, required: true })

const promptTemplateStore = usePromptTemplateStore()
const { templates, loading } = storeToRefs(promptTemplateStore)

const mode = ref<'list' | 'edit' | 'fill'>('list')
const editingId = ref<string | null>(null)
const formName = ref('')
const formDescription = ref('')
const formContent = ref('')
const saving = ref(false)

const fillingTemplate = ref<IPromptTemplate | null>(null)
const fillValues = ref<Record<string, string>>({})

const deleteConfirmDialog = ref(false)
const templateToDelete = ref<IPromptTemplate | null>(null)

watch(isOpen, (open) => {
  if (open && !templates.value.length)
    promptTemplateStore.fetchTemplates()

  if (!open) {
    mode.value = 'list'
    editingId.value = null
    fillingTemplate.value = null
  }
})

function openCreate() {
  mode.value = 'edit'
  editingId.value = null
  formName.value = ''
  formDescription.value = ''
  formContent.value = ''
}

function openEdit(template: IPromptTemplate) {
  mode.value = 'edit'
  editingId.value = template.id
  formName.value = template.name
  formDescription.value = template.description
  formContent.value = template.content
}

function cancelEdit() {
  mode.value = 'list'
  editingId.value = null
}

async function saveForm() {
  const name = formName.value.trim()
  const content = formContent.value.trim()
  if (!name || !content)
    return

  saving.value = true

  if (editingId.value)
    await promptTemplateStore.updateTemplate(editingId.value, name, content, formDescription.value.trim())
  else
    await promptTemplateStore.createTemplate(name, content, formDescription.value.trim())

  saving.value = false
  cancelEdit()
}

function requestDelete(template: IPromptTemplate) {
  templateToDelete.value = template
  deleteConfirmDialog.value = true
}

async function confirmDelete() {
  deleteConfirmDialog.value = false

  if (!templateToDelete.value)
    return

  await promptTemplateStore.deleteTemplate(templateToDelete.value.id)
  templateToDelete.value = null
}

function cancelDelete() {
  deleteConfirmDialog.value = false
  templateToDelete.value = null
}

function useTemplate(template: IPromptTemplate) {
  const variables = extractTemplateVariables(template.content)

  if (!variables.length) {
    emit('insert', template.content)
    isOpen.value = false

    return
  }

  fillingTemplate.value = template
  fillValues.value = Object.fromEntries(variables.map(name => [name, '']))
  mode.value = 'fill'
}

function cancelFill() {
  mode.value = 'list'
  fillingTemplate.value = null
}

function confirmFill() {
  if (!fillingTemplate.value)
    return

  emit('insert', renderTemplate(fillingTemplate.value.content, fillValues.value))
  isOpen.value = false
}
</script>

<template>
  <v-dialog
    v-model="isOpen"
    max-width="520"
    transition="dialog-rise-transition"
  >
    <v-card class="reichat-dialog">
      <div class="reichat-dialog-header">
        <span class="reichat-dialog-title">
          {{ mode === 'list'
            ? 'Prompt templates'
            : mode === 'fill'
              ? fillingTemplate?.name
              : editingId
                ? 'Edit template'
                : 'New template' }}
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
            v-for="template in templates"
            :key="template.id"
            class="template-row"
            @click="useTemplate(template)"
          >
            <div class="template-row-main">
              <span class="template-row-name">{{ template.name }}</span>

              <span class="template-row-desc">{{ template.description || template.content }}</span>
            </div>

            <div class="template-row-actions">
              <button
                type="button"
                class="reichat-dialog-close"
                title="Edit template"
                aria-label="Edit template"
                @click.stop="openEdit(template)"
              >
                <v-icon
                  size="14"
                  icon="$mdiPencil"
                />
              </button>

              <button
                type="button"
                class="reichat-dialog-close template-row-delete"
                title="Delete template"
                aria-label="Delete template"
                @click.stop="requestDelete(template)"
              >
                <v-icon
                  size="14"
                  icon="$mdiDelete"
                />
              </button>
            </div>

            <span class="template-row-use font-mono">Use</span>
          </div>

          <div
            v-if="!loading && !templates.length"
            class="template-empty"
          >
            No saved templates yet — create one below.
          </div>

          <button
            type="button"
            class="dashed-add-row"
            @click="openCreate"
          >
            + New template
          </button>
        </template>

        <template v-else-if="mode === 'edit'">
          <v-text-field
            v-model="formName"
            label="Name"
            density="compact"
            autofocus
            class="mb-2"
          />

          <v-text-field
            v-model="formDescription"
            label="Description (optional)"
            density="compact"
            class="mb-2"
          />

          <v-textarea
            v-model="formContent"
            label="Content"
            hint="Use {{variable}} placeholders for anything you want to fill in each time."
            persistent-hint
            auto-grow
            rows="4"
          />
        </template>

        <template v-else>
          <div class="fill-hint">
            Fill in the blanks, then insert into the message box.
          </div>

          <v-text-field
            v-for="name in Object.keys(fillValues)"
            :key="name"
            v-model="fillValues[name]"
            :label="name"
            density="compact"
            class="mb-2"
          />
        </template>
      </v-card-text>

      <v-card-actions
        v-if="mode !== 'list'"
        class="reichat-dialog-actions"
      >
        <v-btn
          variant="text"
          @click="mode === 'fill'
            ? cancelFill()
            : cancelEdit()"
        >
          {{ mode === 'fill'
            ? 'Back'
            : 'Cancel' }}
        </v-btn>

        <v-spacer />

        <v-btn
          v-if="mode === 'edit'"
          color="mint-btn"
          variant="flat"
          :loading="saving"
          :disabled="!formName.trim() || !formContent.trim()"
          @click="saveForm"
        >
          Save
        </v-btn>

        <v-btn
          v-if="mode === 'fill'"
          color="mint-btn"
          variant="flat"
          @click="confirmFill"
        >
          Insert
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>

  <ConfirmDialog
    v-model="deleteConfirmDialog"
    title="Delete prompt template"
    message="This can't be undone."
    confirm-label="Delete"
    confirm-color="red"
    @confirm="confirmDelete"
    @cancel="cancelDelete"
  />
</template>

<style scoped>
.template-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 9px 11px;
  border-radius: 10px;
  cursor: pointer;
  transition: background-color 0.15s ease;
}

.template-row:hover {
  background: var(--color-soft);
}

.template-row-main {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.template-row-name {
  font-size: 12.5px;
  font-weight: 500;
  color: var(--color-ink);
}

.template-row-desc {
  font-size: 11.5px;
  color: var(--color-ink-2);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.template-row-actions {
  display: flex;
  gap: 2px;
  flex-shrink: 0;
  opacity: 0;
  transition: opacity 0.15s ease;
}

.template-row:hover .template-row-actions {
  opacity: 1;
}

.template-row-delete:hover {
  color: var(--color-red);
}

.template-row-use {
  flex-shrink: 0;
  font-size: 10.5px;
  color: var(--color-mint-deep);
}

.template-empty {
  padding: 16px 4px;
  text-align: center;
  font-size: 12.5px;
  color: var(--color-ink-2);
}

.fill-hint {
  font-size: 12.5px;
  color: var(--color-ink-2);
  margin-bottom: 12px;
}
</style>
