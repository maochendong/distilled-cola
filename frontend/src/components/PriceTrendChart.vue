<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import * as echarts from 'echarts'

const props = defineProps({
  districts: { type: Array, default: () => [] },
  visible: { type: Boolean, default: false },
})

const chartRef = ref(null)
let chartInstance = null

async function fetchAndRender() {
  if (!chartRef.value) return

  try {
    const params = props.districts.length ? `?district=${props.districts.join(',')}` : ''
    const r = await fetch(`/api/data/price_trend${params}`)
    const data = await r.json()
    const trends = data.trends || []

    if (!trends.length) {
      if (chartInstance) chartInstance.dispose()
      return
    }

    const districts = [...new Set(trends.map(t => t.district))]
    const months = [...new Set(trends.map(t => t.month))].sort()

    const series = districts.map(d => ({
      name: d,
      type: 'line',
      smooth: true,
      data: months.map(m => {
        const t = trends.find(t => t.district === d && t.month === m)
        return t ? t.avg_price : null
      }),
    }))

    if (chartInstance) chartInstance.dispose()
    chartInstance = echarts.init(chartRef.value)
    chartInstance.setOption({
      tooltip: { trigger: 'axis' },
      legend: { data: districts, bottom: 0, type: 'scroll' },
      grid: { left: 55, right: 20, bottom: 50, top: 20 },
      xAxis: { type: 'category', data: months, axisLabel: { rotate: 30 } },
      yAxis: { type: 'value', name: '元/㎡', axisLabel: { formatter: v => (v / 10000).toFixed(1) + '万' } },
      series,
      color: ['#2563EB', '#7C3AED', '#059669', '#D97706', '#DC2626', '#0891B2'],
    })
  } catch (e) { /* ignore */ }
}

watch(() => props.districts, fetchAndRender)

onMounted(fetchAndRender)
onUnmounted(() => { if (chartInstance) chartInstance.dispose() })
</script>

<template>
  <div v-if="visible" class="chart-overlay" @click.self="$emit('close')">
    <div class="chart-card">
      <div class="chart-header">
        <span class="chart-title">📈 价格趋势</span>
        <button class="chart-close" @click="$emit('close')">✕</button>
      </div>
      <div ref="chartRef" class="chart-container"></div>
    </div>
  </div>
</template>

<style scoped>
.chart-overlay {
  position: fixed; inset: 0; background: rgba(0,0,0,0.3);
  display: flex; align-items: center; justify-content: center;
  z-index: 100; padding: 1rem;
}
.chart-card {
  background: white; border-radius: 16px; width: 100%; max-width: 700px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.12);
}
.chart-header {
  display: flex; justify-content: space-between; align-items: center;
  padding: 1rem 1.2rem; border-bottom: 1px solid var(--border);
}
.chart-title { font-size: 1.05rem; font-weight: 700; }
.chart-close { background: none; border: none; font-size: 1.2rem; cursor: pointer; color: var(--text-secondary); }
.chart-container { width: 100%; height: 400px; padding: 0.5rem; }
</style>
