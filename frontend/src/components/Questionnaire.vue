<script setup>
import { ref, computed, onMounted } from 'vue'

const props = defineProps({
  identityType: { type: String, default: 'general' },
  profileId: { type: String, default: '' },
})

const emit = defineEmits(['complete', 'close'])

const API = '/api'
const questions = ref([])
const currentStep = ref(0)
const answers = ref({})
const submitting = ref(false)

const allTemplates = ref({})

onMounted(async () => {
  try {
    const r = await fetch(`${API}/questionnaire/templates`)
    const data = await r.json()
    allTemplates.value = data.templates || {}
    questions.value = allTemplates.value[props.identityType] || []
  } catch (e) {
    questions.value = []
  }
})

const totalSteps = computed(() => questions.value.length)
const currentQ = computed(() => questions.value[currentStep.value])

const isLast = computed(() => currentStep.value >= totalSteps.value - 1)
const progress = computed(() => totalSteps.value > 0 ? ((currentStep.value + 1) / totalSteps.value) * 100 : 0)

function next() {
  if (isLast.value) submit()
  else currentStep.value++
}

function skip() {
  if (isLast.value) submit()
  else currentStep.value++
}

async function submit() {
  submitting.value = true
  try {
    await fetch(`${API}/questionnaire`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        profile_id: props.profileId,
        identity_type: props.identityType,
        answers: answers.value,
      }),
    })
  } catch (e) { /* ignore */ }
  submitting.value = false
  emit('complete', answers.value)
}
</script>

<template>
  <div v-if="questions.length > 0" class="questionnaire-overlay" @click.self="emit('close')">
    <div class="questionnaire-card">
      <div class="q-header">
        <span class="q-title">了解你的需求</span>
        <button class="q-close" @click="emit('close')">✕</button>
      </div>

      <div class="progress-bar">
        <div class="progress-fill" :style="{ width: progress + '%' }"></div>
      </div>
      <div class="step-indicator">{{ currentStep + 1 }} / {{ totalSteps }}</div>

      <div class="q-body" v-if="currentQ">
        <label class="q-label">{{ currentQ.label }}</label>

        <input
          v-if="currentQ.type === 'text'"
          v-model="answers[currentQ.id]"
          class="q-input"
          :placeholder="currentQ.placeholder"
          @keyup.enter="next"
        />

        <div v-if="currentQ.type === 'radio'" class="q-options">
          <button
            v-for="opt in currentQ.options"
            :key="opt"
            :class="['q-option', { selected: answers[currentQ.id] === opt }]"
            @click="answers[currentQ.id] = opt; next()"
          >{{ opt }}</button>
        </div>

        <input
          v-if="currentQ.type === 'number'"
          v-model.number="answers[currentQ.id]"
          type="number"
          class="q-input"
          :placeholder="currentQ.placeholder"
          @keyup.enter="next"
        />

        <div class="q-actions">
          <button v-if="!isLast" class="q-skip" @click="skip">跳过</button>
          <button class="q-next" @click="next" :disabled="submitting">
            {{ isLast ? '完成' : '下一步' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.questionnaire-overlay {
  position: fixed; inset: 0; background: rgba(0,0,0,0.3);
  display: flex; align-items: center; justify-content: center;
  z-index: 100; padding: 1rem;
}
.questionnaire-card {
  background: white; border-radius: 16px; width: 100%; max-width: 440px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.12);
}
.q-header {
  display: flex; justify-content: space-between; align-items: center;
  padding: 1rem 1.2rem; border-bottom: 1px solid var(--border);
}
.q-title { font-size: 1.05rem; font-weight: 700; }
.q-close { background: none; border: none; font-size: 1.2rem; cursor: pointer; color: var(--text-secondary); }

.progress-bar { height: 4px; background: var(--border); margin: 0 1.2rem; border-radius: 2px; }
.progress-fill { height: 100%; background: var(--primary); border-radius: 2px; transition: width 0.3s; }
.step-indicator { text-align: right; font-size: 0.78rem; color: var(--text-secondary); padding: 0.3rem 1.2rem 0; }

.q-body { padding: 1.2rem; }
.q-label { font-size: 1.1rem; font-weight: 600; display: block; margin-bottom: 1rem; }
.q-input {
  width: 100%; padding: 0.75rem 1rem; font-size: 1rem;
  border: 2px solid var(--border); border-radius: 10px; outline: none;
}
.q-input:focus { border-color: var(--primary); }
.q-options { display: flex; flex-direction: column; gap: 0.5rem; }
.q-option {
  padding: 0.8rem 1rem; font-size: 0.95rem; border: 2px solid var(--border);
  border-radius: 10px; background: white; cursor: pointer; text-align: left;
  transition: all 0.15s;
}
.q-option:hover { border-color: var(--primary); }
.q-option.selected { border-color: var(--primary); background: var(--primary-light); color: var(--primary); font-weight: 500; }

.q-actions { display: flex; gap: 0.5rem; justify-content: flex-end; margin-top: 1.2rem; }
.q-skip { padding: 0.6rem 1rem; font-size: 0.88rem; background: none; border: 1px solid var(--border); border-radius: 8px; cursor: pointer; color: var(--text-secondary); }
.q-next { padding: 0.6rem 1.2rem; font-size: 0.88rem; background: var(--primary); color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: 600; }
.q-next:disabled { opacity: 0.5; }
</style>
