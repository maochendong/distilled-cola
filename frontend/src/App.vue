<script setup>
import { ref, nextTick } from 'vue'
import { marked } from 'marked'

marked.setOptions({ breaks: true, gfm: true })

const API = '/api'
const messages = ref([])
const loading = ref(false)
const showHome = ref(true)
const query = ref('')
const topK = ref(5)

function renderMarkdown(text) {
  return marked(text)
}

async function ask() {
  const q = query.value.trim()
  if (!q || loading.value) return
  loading.value = true
  showHome.value = false
  messages.value.push({ role: 'user', content: q })
  query.value = ''
  await nextTick()
  scrollBottom()

  const msg = { role: 'assistant', content: '', confidence: null, sources: [], reasoning_chains_used: 0 }
  messages.value.push(msg)

  const url = `${API}/ask/stream?query=${encodeURIComponent(q)}&top_k=${topK.value}`
  const eventSource = new EventSource(url)

  eventSource.onmessage = (e) => {
    const chunk = JSON.parse(e.data)
    if (chunk.type === 'token') {
      msg.content += chunk.content
    } else if (chunk.type === 'done') {
      msg.confidence = chunk.confidence
      // Fetch sources via blocking ask for now
      eventSource.close()
      loading.value = false
      loadSources(q, msg)
    } else if (chunk.type === 'error' || chunk.type === 'warning') {
      msg.content += chunk.content
    }
  }

  eventSource.onerror = () => {
    eventSource.close()
    loading.value = false
    loadSources(q, msg)
  }
}

function loadSources(q, msg) {
  fetch(`${API}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: q, top_k: topK.value }),
  })
  .then(r => r.json())
  .then(data => {
    msg.sources = data.sources || []
    msg.reasoning_chains_used = data.reasoning_chains_used || 0
    msg.web_search_used = data.web_search_used || false
  })
  .catch(() => {})
}

function scrollBottom() {
  nextTick(() => {
    const el = document.getElementById('chat-end')
    if (el) el.scrollIntoView({ behavior: 'smooth' })
  })
}

function newChat() {
  messages.value = []
  showHome.value = true
  loading.value = false
}
</script>

<template>
  <div class="app">
    <!-- ═══ Home ═══ -->
    <div v-if="showHome" class="home">
      <div class="spacer"></div>
      <img src="/logo.jpeg" alt="蒸馏小可乐" class="logo" />
      <h1 class="title">🏠 蒸馏小可乐</h1>
      <p class="tagline">上海房产分析专家 — 基于博主知识蒸馏的四步推演系统</p>
      <div class="search-box">
        <input
          v-model="query"
          type="text"
          placeholder="输入你的上海房产问题，如：800万预算前滩vs大宁怎么选？"
          class="search-input"
          @keyup.enter="ask()"
        />
        <select v-model="topK" class="topk-select">
          <option v-for="k in [1,3,5,8,10]" :key="k" :value="k">{{ k }}</option>
        </select>
        <button class="search-btn" @click="ask()" :disabled="loading">🔍 分析</button>
      </div>
      <p class="topk-label">检索数量</p>
    </div>

    <!-- ═══ Chat ═══ -->
    <div v-if="!showHome" class="chat">
      <div class="chat-header">
        <span class="chat-title">🏠 蒸馏小可乐</span>
        <button class="new-chat-btn" @click="newChat">＋ 新对话</button>
      </div>
      <div class="chat-messages">
        <div v-for="(msg, i) in messages" :key="i" :class="['msg', msg.role]">
          <div class="msg-content" v-html="renderMarkdown(msg.content)"></div>

          <!-- Metrics (T-009) -->
          <div v-if="msg.role === 'assistant' && msg.confidence !== null" class="metrics">
            <span>🎯 置信度 {{ (msg.confidence * 100).toFixed(0) }}%</span>
            <span>🧠 推理链引用 {{ msg.reasoning_chains_used }}</span>
            <span>📚 参考来源 {{ msg.sources?.length || 0 }}</span>
            <span v-if="msg.web_search_used">🌐 实时搜索</span>
          </div>

          <!-- 参考来源 (T-009) -->
          <details v-if="msg.role === 'assistant' && msg.sources?.length" class="sources">
            <summary>📚 参考来源</summary>
            <div v-for="(s, si) in msg.sources" :key="si" class="source-item">
              <strong>{{ s.source || s.metadata?.source || '未知' }}</strong>
              <span class="source-score">(相关度: {{ (s.score || s.rerank_score || 0).toFixed(2) }})</span>
              <p class="source-snippet">{{ s.snippet || s.text || s.content || '' }}</p>
            </div>
          </details>
        </div>
        <div id="chat-end"></div>
      </div>

      <div class="chat-input-bar">
        <input v-model="query" type="text" placeholder="继续提问..." class="chat-input" @keyup.enter="ask()" />
        <select v-model="topK" class="topk-select-small">
          <option v-for="k in [1,3,5,8,10]" :key="k" :value="k">{{ k }}</option>
        </select>
        <button class="send-btn" @click="ask()" :disabled="loading">发送</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.app { min-height: 100vh; }

/* ── Home ── */
.home { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 70vh; }
.spacer { height: 8vh; }
.logo { width: 120px; margin-bottom: 1.5rem; }
.title { font-size: 2.5rem; font-weight: 700; color: var(--text); margin-bottom: 0.5rem; }
.tagline { font-size: 1.05rem; color: var(--text-secondary); margin-bottom: 2rem; }
.search-box { display: flex; width: 100%; max-width: 640px; gap: 0.4rem; align-items: center; }
.search-input { flex: 1; padding: 0.9rem 1.2rem; font-size: 1rem; border: 2px solid var(--border); border-radius: var(--radius); outline: none; transition: border-color 0.2s; }
.search-input:focus { border-color: var(--primary); }
.topk-select { padding: 0.9rem 0.5rem; font-size: 0.9rem; border: 2px solid var(--border); border-radius: var(--radius); background: white; cursor: pointer; min-width: 48px; }
.topk-select-small { padding: 0.7rem 0.4rem; font-size: 0.85rem; border: 2px solid var(--border); border-radius: 8px; background: white; cursor: pointer; min-width: 44px; }
.search-btn { padding: 0.9rem 1.5rem; font-size: 1rem; background: var(--primary); color: #fff; border: none; border-radius: var(--radius); cursor: pointer; font-weight: 600; white-space: nowrap; }
.search-btn:hover { background: var(--primary-dark); }
.search-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.topk-label { font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.3rem; }

/* ── Chat ── */
.chat { max-width: 800px; margin: 0 auto; padding: 1rem; min-height: 100vh; display: flex; flex-direction: column; }
.chat-header { display: flex; justify-content: space-between; align-items: center; padding: 0.75rem 0; border-bottom: 1px solid var(--border); margin-bottom: 1rem; }
.chat-title { font-size: 1.1rem; font-weight: 700; }
.new-chat-btn { padding: 0.4rem 0.9rem; font-size: 0.85rem; background: none; border: 1px solid var(--border); border-radius: 8px; cursor: pointer; color: var(--text-secondary); }
.new-chat-btn:hover { background: var(--primary-light); color: var(--primary); }

.chat-messages { flex: 1; overflow-y: auto; padding-bottom: 1rem; }
.msg { margin-bottom: 1.2rem; }
.msg.user .msg-content { background: var(--primary-light); color: var(--primary-dark); margin-left: auto; max-width: 80%; padding: 0.8rem 1rem; border-radius: var(--radius); border-bottom-right-radius: 4px; }
.msg.assistant .msg-content { background: var(--white); border: 1px solid var(--border); max-width: 100%; padding: 1rem 1.5rem; border-radius: var(--radius); border-bottom-left-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); font-size: 0.95rem; line-height: 1.8; }
.msg-content :deep(h1) { font-size: 1.5rem; margin: 1rem 0 0.5rem; }
.msg-content :deep(h2) { font-size: 1.25rem; margin: 0.8rem 0 0.4rem; }
.msg-content :deep(h3) { font-size: 1.1rem; margin: 0.6rem 0 0.3rem; }
.msg-content :deep(p) { margin: 0.5rem 0; }
.msg-content :deep(ul), .msg-content :deep(ol) { padding-left: 1.5rem; margin: 0.5rem 0; }
.msg-content :deep(li) { margin: 0.25rem 0; }
.msg-content :deep(code) { background: #F1F5F9; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.88em; }
.msg-content :deep(pre) { background: #1E293B; color: #E2E8F0; padding: 1rem; border-radius: 8px; overflow-x: auto; margin: 0.8rem 0; }
.msg-content :deep(pre code) { background: none; padding: 0; color: inherit; }
.msg-content :deep(table) { border-collapse: collapse; width: 100%; margin: 0.8rem 0; }
.msg-content :deep(th), .msg-content :deep(td) { border: 1px solid var(--border); padding: 0.5rem 0.8rem; text-align: left; }
.msg-content :deep(th) { background: #F8FAFC; font-weight: 600; }
.msg-content :deep(blockquote) { border-left: 3px solid var(--primary); padding: 0.3rem 1rem; margin: 0.8rem 0; background: var(--primary-light); border-radius: 0 8px 8px 0; }
.msg-content :deep(strong) { font-weight: 700; }
.msg-content :deep(hr) { border: none; border-top: 1px solid var(--border); margin: 1rem 0; }

/* Metrics */
.metrics { display: flex; gap: 1rem; flex-wrap: wrap; font-size: 0.82rem; color: var(--text-secondary); margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid var(--border); }
.metrics span { background: var(--primary-light); padding: 0.15rem 0.6rem; border-radius: 6px; }

/* Sources */
.sources { margin-top: 0.6rem; font-size: 0.88rem; }
.sources summary { cursor: pointer; color: var(--primary); font-weight: 500; padding: 0.4rem 0; }
.source-item { padding: 0.5rem 0; border-bottom: 1px solid var(--border); }
.source-item:last-child { border-bottom: none; }
.source-score { color: var(--text-secondary); font-size: 0.8rem; margin-left: 0.4rem; }
.source-snippet { color: var(--text-secondary); font-size: 0.82rem; margin: 0.2rem 0 0; font-style: italic; }

.chat-input-bar { display: flex; gap: 0.5rem; padding: 1rem 0; border-top: 1px solid var(--border); background: rgba(255,255,255,0.8); backdrop-filter: blur(8px); position: sticky; bottom: 0; }
.chat-input { flex: 1; padding: 0.8rem 1rem; font-size: 0.95rem; border: 2px solid var(--border); border-radius: var(--radius); outline: none; }
.chat-input:focus { border-color: var(--primary); }
.send-btn { padding: 0.8rem 1.2rem; background: var(--primary); color: #fff; border: none; border-radius: var(--radius); cursor: pointer; font-weight: 600; }
.send-btn:hover { background: var(--primary-dark); }
.send-btn:disabled { opacity: 0.5; cursor: not-allowed; }

@media (max-width: 768px) {
  .title { font-size: 1.8rem; }
  .search-box { flex-direction: column; }
  .search-btn { width: 100%; }
  .chat { padding: 0.5rem; }
  .msg.user .msg-content { max-width: 90%; }
  .msg.assistant .msg-content { max-width: 100%; padding: 0.8rem 1rem; }
}
</style>
