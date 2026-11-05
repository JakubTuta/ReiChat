<script setup lang="ts">
const isOpen = defineModel<boolean>('isOpen', { default: false, required: true })
const format = defineModel<{
  field: string
  type: string
  description?: string
  arrayType?: string
}[]>('format', { default: () => [], required: true })
const isFormValid = defineModel<boolean>('isFormValid', { default: () => false, required: true })
const enforced = defineModel<boolean>('enforced', { default: () => false, required: true })

const { form, isValid } = useForm()

const simpleTypes = [
  { value: 'string', title: 'String' },
  { value: 'number', title: 'Number' },
  { value: 'boolean', title: 'Boolean' },
  { value: 'date', title: 'Date' },
]

const possibleTypes = [
  ...simpleTypes,
  { value: 'array', title: 'Array' },
]

const canEnforce = computed(() => isFormValid.value && format.value.length > 0)

function addNewType() {
  format.value.push({ field: '', type: '' })
}

function removeType(index: number) {
  format.value.splice(index, 1)
}

async function validate() {
  isFormValid.value = await isValid()

  if (!canEnforce.value)
    enforced.value = false
}

watch(isOpen, (open) => {
  if (!open)
    validate()
})

function downloadJSON() {
  const filename = 'structured_output.json'
  const jsonString = JSON.stringify(format.value, null, 2)

  const blob = new Blob([jsonString], { type: 'application/json' })

  const url = URL.createObjectURL(blob)

  const link = document.createElement('a')
  link.href = url
  link.download = filename

  document.body.appendChild(link)
  link.click()

  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

function validateFormatItem(item: any): boolean {
  if (!item.field || !item.type)
    return false

  const validTypes = possibleTypes.map(type => type.value)
  if (!validTypes.includes(item.type))
    return false

  if (item.type === 'array') {
    const validArrayTypes = simpleTypes.map(type => type.value)

    return item.arrayType && validArrayTypes.includes(item.arrayType)
  }

  if (item.type !== 'array' && item.arrayType !== undefined)
    delete item.arrayType

  return true
}

function importJSON() {
  const input = document.createElement('input')
  input.type = 'file'
  input.accept = 'application/json'

  input.addEventListener('change', (event) => {
    const file = (event.target as HTMLInputElement).files?.[0]
    if (!file)
      return

    const reader = new FileReader()
    reader.onload = (e) => {
      try {
        const json = JSON.parse(e.target?.result as string)

        const validData = json.filter(validateFormatItem)

        format.value = validData
        validate()
      }
      catch (error) {
        console.error('Invalid JSON file:', error)
      }
    }
    reader.readAsText(file)
  })

  input.click()
}
</script>

<template>
  <v-dialog
    v-model="isOpen"
    max-width="540"
    transition="dialog-rise-transition"
  >
    <v-card class="reichat-dialog">
      <div class="reichat-dialog-header">
        <span class="reichat-dialog-title">JSON output</span>

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
        <v-form
          ref="form"
          validate-on="eager"
          @update:model-value="validate"
        >
          <div
            v-for="(formatLine, index) in format"
            :key="index"
            class="field-row"
          >
            <div class="field-row-inputs">
              <v-text-field
                v-model="formatLine.field"
                density="compact"
                :rules="[requiredRule()]"
                label="Field"
              />

              <v-select
                v-model="formatLine.type"
                :items="possibleTypes"
                density="compact"
                :rules="[requiredRule()]"
                label="Type"
              />

              <button
                type="button"
                class="reichat-dialog-close field-row-remove"
                title="Remove field"
                aria-label="Remove field"
                @click="removeType(index)"
              >
                <v-icon
                  size="16"
                  icon="$mdiDelete"
                />
              </button>
            </div>

            <v-select
              v-if="formatLine.type === 'array'"
              v-model="formatLine.arrayType"
              :items="simpleTypes"
              density="compact"
              label="Array type"
              class="mb-2"
            />

            <v-textarea
              v-model="formatLine.description"
              density="compact"
              label="Description"
              rows="1"
              auto-grow
              hint="(Optional) Anything that helps the model understand this field."
            />
          </div>

          <button
            type="button"
            class="dashed-add-row"
            @click="addNewType"
          >
            + Add field
          </button>
        </v-form>

        <div class="enforce-row">
          <div>
            <span class="enforce-label">Enforce this schema</span>

            <span
              class="enforce-hint"
              :class="{'enforce-hint--amber': !canEnforce}"
            >
              {{ canEnforce
                ? 'Responses must match the fields above'
                : 'Name at least one field to enable' }}
            </span>
          </div>

          <v-switch
            v-model="enforced"
            :disabled="!canEnforce"
            density="compact"
            hide-details
            color="mint-btn"
          />
        </div>
      </v-card-text>

      <v-card-actions class="reichat-dialog-actions">
        <v-btn
          variant="text"
          prepend-icon="$mdiUpload"
          @click="importJSON()"
        >
          Import
        </v-btn>

        <v-btn
          variant="text"
          prepend-icon="$mdiDownload"
          @click="downloadJSON()"
        >
          Export
        </v-btn>

        <v-spacer />

        <v-btn
          color="mint-btn"
          variant="flat"
          @click="isOpen = false"
        >
          Done
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<style scoped>
.field-row {
  padding: 10px 0;
  border-bottom: 1px solid var(--color-line-2);
  margin-bottom: 8px;
}

.field-row-inputs {
  display: flex;
  align-items: flex-start;
  gap: 8px;
}

.field-row-inputs > :first-child,
.field-row-inputs > :nth-child(2) {
  flex: 1;
  min-width: 0;
}

.field-row-remove {
  margin-top: 6px;
  color: var(--color-red);
}

.enforce-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-top: 16px;
  padding-top: 14px;
  border-top: 1px solid var(--color-line-2);
}

.enforce-label {
  display: block;
  font-size: 13px;
  font-weight: 500;
  color: var(--color-ink);
}

.enforce-hint {
  display: block;
  font-size: 11px;
  color: var(--color-ink-3);
}

.enforce-hint--amber {
  color: var(--color-amber);
}
</style>
