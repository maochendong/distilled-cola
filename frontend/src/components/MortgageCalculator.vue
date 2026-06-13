<script setup>
import { ref, computed } from 'vue'

const props = defineProps({
  visible: { type: Boolean, default: false },
})

const emit = defineEmits(['close'])

const totalPrice = ref(500)
const downRatio = ref(30)
const years = ref(30)
const rate = ref(3.15)
const firstHome = ref(true)
const area = ref(90)

const loanPrincipal = computed(() => totalPrice.value * (1 - downRatio.value / 100))
const monthlyRate = computed(() => rate.value / 12 / 100)
const months = computed(() => years.value * 12)

const monthlyPayment = computed(() => {
  const p = loanPrincipal.value * 10000
  const r = monthlyRate.value
  const n = months.value
  if (r === 0) return p / n
  return (p * r * Math.pow(1 + r, n)) / (Math.pow(1 + r, n) - 1)
})

const totalPayment = computed(() => monthlyPayment.value * months.value)
const totalInterest = computed(() => totalPayment.value - loanPrincipal.value * 10000)

const downPayment = computed(() => totalPrice.value * downRatio.value / 100)

const deedTax = computed(() => {
  const p = totalPrice.value * 10000
  if (!firstHome.value) return p * 0.03
  if (area.value <= 90) return p * 0.01
  return p * 0.015
})

const totalCost = computed(() => totalPrice.value * 10000 + totalInterest.value + deedTax.value)

function formatYuan(v) {
  if (v >= 10000) return (v / 10000).toFixed(0) + '万'
  if (v >= 1000) return v.toFixed(0) + '元'
  return v.toFixed(0) + '元'
}

function formatWan(v) {
  return v.toFixed(0) + '万'
}
</script>

<template>
  <div v-if="visible" class="calculator-overlay" @click.self="emit('close')">
    <div class="calculator-card">
      <div class="calc-header">
        <span class="calc-title">🏠 房贷计算器</span>
        <button class="calc-close" @click="emit('close')">✕</button>
      </div>

      <div class="calc-body">
        <div class="form-grid">
          <div class="form-group">
            <label>总价（万）</label>
            <input v-model.number="totalPrice" type="range" min="100" max="3000" step="10" />
            <span class="input-value">{{ totalPrice }}万</span>
          </div>

          <div class="form-group">
            <label>首付比例</label>
            <input v-model.number="downRatio" type="range" min="15" max="80" step="5" />
            <span class="input-value">{{ downRatio }}%</span>
          </div>

          <div class="form-group">
            <label>贷款年限</label>
            <input v-model.number="years" type="range" min="5" max="30" step="5" />
            <span class="input-value">{{ years }}年</span>
          </div>

          <div class="form-group">
            <label>商贷利率（%）</label>
            <input v-model.number="rate" type="range" min="2.5" max="5" step="0.05" />
            <span class="input-value">{{ rate }}%</span>
          </div>

          <div class="form-group half">
            <label>面积（㎡）</label>
            <input v-model.number="area" type="range" min="30" max="300" step="5" />
            <span class="input-value">{{ area }}㎡</span>
          </div>

          <div class="form-group half">
            <label class="checkbox-label">
              <input v-model="firstHome" type="checkbox" /> 首套房
            </label>
          </div>
        </div>

        <div class="results-card">
          <div class="result-row main">
            <span class="result-label">月供</span>
            <span class="result-value highlight">{{ formatYuan(monthlyPayment) }} /月</span>
          </div>
          <div class="result-row">
            <span class="result-label">贷款总额</span>
            <span class="result-value">{{ formatWan(loanPrincipal) }}</span>
          </div>
          <div class="result-row">
            <span class="result-label">利息总额</span>
            <span class="result-value">{{ formatYuan(totalInterest) }}</span>
          </div>
          <div class="result-row">
            <span class="result-label">首付</span>
            <span class="result-value">{{ formatWan(downPayment) }}</span>
          </div>
          <div class="result-row">
            <span class="result-label">契税</span>
            <span class="result-value">{{ formatYuan(deedTax) }}</span>
          </div>
          <div class="result-row total">
            <span class="result-label">购房总成本</span>
            <span class="result-value">{{ formatYuan(totalCost) }}</span>
          </div>
          <div class="progress-bar">
            <div class="progress-down" :style="{ flex: downRatio }"></div>
            <div class="progress-loan" :style="{ flex: 100 - downRatio }"></div>
          </div>
          <div class="progress-labels">
            <span>首付 {{ formatWan(downPayment) }}</span>
            <span>贷款 {{ formatWan(loanPrincipal) }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.calculator-overlay {
  position: fixed; inset: 0; background: rgba(0,0,0,0.3);
  display: flex; align-items: center; justify-content: center;
  z-index: 100; padding: 1rem;
}
.calculator-card {
  background: white; border-radius: 16px; width: 100%; max-width: 480px;
  max-height: 90vh; overflow-y: auto; box-shadow: 0 8px 32px rgba(0,0,0,0.12);
}
.calc-header {
  display: flex; justify-content: space-between; align-items: center;
  padding: 1rem 1.2rem; border-bottom: 1px solid var(--border);
}
.calc-title { font-size: 1.05rem; font-weight: 700; }
.calc-close { background: none; border: none; font-size: 1.2rem; cursor: pointer; color: var(--text-secondary); }
.calc-body { padding: 1.2rem; }
.form-grid { display: flex; flex-direction: column; gap: 0.8rem; }
.form-group label { font-size: 0.85rem; color: var(--text-secondary); display: block; margin-bottom: 0.2rem; }
.form-group input[type="range"] { width: 100%; accent-color: var(--primary); }
.input-value { font-size: 0.82rem; color: var(--primary); font-weight: 600; }
.form-group.half { display: flex; gap: 1rem; align-items: center; }
.checkbox-label { display: flex; align-items: center; gap: 0.4rem; cursor: pointer; }
.checkbox-label input { width: auto; accent-color: var(--primary); }

.results-card {
  margin-top: 1.2rem; background: #F8FAFC; border-radius: 12px;
  padding: 1rem; border: 1px solid var(--border);
}
.result-row { display: flex; justify-content: space-between; padding: 0.4rem 0; }
.result-row.main { padding: 0.6rem 0; }
.result-row.total { border-top: 1px solid var(--border); margin-top: 0.4rem; padding-top: 0.6rem; font-weight: 600; }
.result-label { font-size: 0.88rem; color: var(--text-secondary); }
.result-value { font-size: 0.88rem; font-weight: 500; }
.result-value.highlight { font-size: 1.1rem; font-weight: 700; color: var(--primary); }

.progress-bar { display: flex; height: 8px; border-radius: 4px; overflow: hidden; margin-top: 0.6rem; }
.progress-down { background: var(--primary); }
.progress-loan { background: #DBEAFE; }
.progress-labels { display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.3rem; }
</style>
