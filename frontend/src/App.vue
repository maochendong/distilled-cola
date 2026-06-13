<script setup>
import { ref, nextTick, onMounted } from 'vue'
import { marked } from 'marked'

marked.setOptions({ breaks: true, gfm: true })

const API = '/api'
const messages = ref([])
const loading = ref(false)
const streaming = ref(false)
const showHome = ref(true)
const query = ref('')
const convId = ref(null)
const topK = 5
const identity = ref('')
const showIdentity = ref(false)

const identityOptions = [
  { value: '', label: '通用模式', desc: '标准分析' },
  { value: '你是一位首套刚需购房者，重点关注通勤便利、总价门槛、首付月供、增值潜力。', label: '首套刚需', desc: '侧重通勤与上车门槛' },
  { value: '你是一位独立女性购房者，重点关注安全底线、产权规划、长期流动性。', label: '独立女性', desc: '侧重安全与产权' },
  { value: '你是一位有家庭的改善型购房者，重点关注学区、空间动线、置换链条、社区儿童友好度。', label: '家庭改善', desc: '侧重学区与置换' },
  { value: '你是一位养老置业者，重点关注医疗保障、无障碍通道、社区支持系统。', label: '养老置业', desc: '侧重医疗与适老' },
]

function buildQuery(raw) {
  if (!identity.value) return raw
  return `[背景] ${identity.value}\n\n问题: ${raw}`
}

const exampleQuestions = [
  '总价500万，上海哪里性价比最高？',
  '800万预算改善，浦东和浦西怎么选？',
  '看中一套房，怎么判断价格合不合理？',
  '现在该卖房吗，还是再等等？',
  '上海现在买房，选新房还是二手次新？',
  '300万预算首套上车，通勤和增值哪个更重要？',
]

function renderMarkdown(text) {
  return marked(text)
}

async function ask() {
  const raw = query.value.trim()
  if (!raw || loading.value) return

  const q = buildQuery(raw)

  loading.value = true
  streaming.value = true
  showHome.value = false

  messages.value.push({ role: 'user', content: raw })
  query.value = ''
  await nextTick()
  scrollBottom()

  const msg = ref({ role: 'assistant', content: '', confidence: null, sources: [], reasoning_chains_used: 0, web_search_used: false })
  messages.value.push(msg.value)

  const params = new URLSearchParams({ query: q, top_k: String(topK) })
  if (convId.value) params.set('conv_id', convId.value)

  const eventSource = new EventSource(`${API}/ask/stream?${params}`)

  eventSource.onmessage = (e) => {
    const chunk = JSON.parse(e.data)
    if (chunk.type === 'token') {
      msg.value.content += chunk.content
    } else if (chunk.type === 'done') {
      msg.value.confidence = chunk.confidence
      msg.value.sources = chunk.sources || []
      msg.value.reasoning_chains_used = chunk.reasoning_chains_used || 0
      msg.value.web_search_used = chunk.web_search_used || false
      convId.value = chunk.conv_id || convId.value
      eventSource.close()
      loading.value = false
      streaming.value = false
    } else if (chunk.type === 'warning') {
      msg.value.content += '\n\n' + chunk.content
    } else if (chunk.type === 'error') {
      msg.value.content += '\n\n' + chunk.content
      loading.value = false
      streaming.value = false
    }
  }

  eventSource.onerror = () => {
    eventSource.close()
    loading.value = false
    streaming.value = false
  }
}

function sendFeedback(aid, rating) {
  fetch(`${API}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ answer_id: aid, rating }),
  }).catch(() => {})
}

function newChat() {
  messages.value = []
  showHome.value = true
  loading.value = false
  streaming.value = false
  convId.value = null
}

function pickExample(q) {
  query.value = q
  ask()
}

function scrollBottom() {
  nextTick(() => {
    const el = document.getElementById('chat-end')
    if (el) el.scrollIntoView({ behavior: 'smooth' })
  })
}
</script>

<template>
  <div class="app">
    <!-- ═══ Home ═══ -->
    <div v-if="showHome" class="home">
      <div class="spacer"></div>
      <img src="/logo.jpeg" alt="蒸馏小可乐" class="logo" />
      <h1 class="title">蒸馏小可乐</h1>
      <p class="tagline">助你决策人生最大一笔投资</p>
      <div class="search-box">
        <input
          v-model="query"
          type="text"
          placeholder="输入你的上海房产问题..."
          class="search-input"
          @keyup.enter="ask()"
        />
        <button class="search-btn" @click="ask()" :disabled="loading">🔍 分析</button>
      </div>
      <div class="identity-bar-home">
        <span class="identity-label">购房身份：</span>
        <button
          v-for="opt in identityOptions"
          :key="opt.value"
          :class="['identity-chip', { active: identity === opt.value }]"
          @click="identity = opt.value"
          :title="opt.desc"
        >{{ opt.label }}</button>
      </div>
      <div class="examples">
        <p class="examples-label">试试这些问题：</p>
        <div class="example-grid">
          <button
            v-for="(eq, i) in exampleQuestions"
            :key="i"
            class="example-btn"
            @click="pickExample(eq)"
          >{{ eq }}</button>
        </div>
      </div>
    </div>

    <!-- ═══ Chat ═══ -->
    <div v-if="!showHome" class="chat">
      <div class="chat-header">
        <span class="chat-title">蒸馏小可乐</span>
        <div class="header-right">
          <div class="identity-selector-compact">
            <button class="identity-toggle" @click="showIdentity = !showIdentity">
              {{ identityOptions.find(o => o.value === identity)?.label || '通用' }}
            </button>
            <div v-if="showIdentity" class="identity-dropdown">
              <button
                v-for="opt in identityOptions"
                :key="opt.value"
                :class="['id-option', { active: identity === opt.value }]"
                @click="identity = opt.value; showIdentity = false"
              >
                <span class="id-label">{{ opt.label }}</span>
                <span class="id-desc">{{ opt.desc }}</span>
              </button>
            </div>
          </div>
          <button class="new-chat-btn" @click="newChat">+ 新对话</button>
        </div>
      </div>

      <div class="chat-messages">
        <div v-for="(msg, i) in messages" :key="i" :class="['msg', msg.role]">
          <div class="msg-label">{{ msg.role === 'user' ? '你' : '分析师' }}</div>
          <div v-if="msg.content" class="msg-content" v-html="renderMarkdown(msg.content)"></div>
          <div v-else class="msg-content streaming-placeholder">
            <span class="dot-pulse"></span>
            <span class="streaming-text">正在分析...</span>
          </div>

          <!-- Metrics -->
          <div v-if="msg.role === 'assistant' && msg.confidence !== null" class="metrics">
            <span class="metric">置信度 {{ (msg.confidence * 100).toFixed(0) }}%</span>
            <span class="metric" v-if="msg.reasoning_chains_used">推理链 {{ msg.reasoning_chains_used }}</span>
            <span class="metric">来源 {{ msg.sources?.length || 0 }}</span>
            <span class="metric web" v-if="msg.web_search_used">实时行情</span>
          </div>

          <!-- Feedback -->
          <div v-if="msg.role === 'assistant' && msg.answer_id" class="feedback-row">
            <button class="fb-btn" @click="sendFeedback(msg.answer_id, 'up')" title="有用">👍</button>
            <button class="fb-btn" @click="sendFeedback(msg.answer_id, 'down')" title="没用">👎</button>
          </div>

          <!-- Sources -->
          <details v-if="msg.role === 'assistant' && msg.sources?.length" class="sources">
            <summary>参考来源 ({{ msg.sources.length }})</summary>
            <div v-for="(s, si) in msg.sources" :key="si" class="source-item">
              <strong>{{ s.source || '未知' }}</strong>
              <span class="source-score">({{ (s.score || 0).toFixed(2) }})</span>
              <span class="source-type" v-if="s.type === 'web'">[实时]</span>
              <p class="source-snippet">{{ s.snippet || '' }}</p>
            </div>
          </details>
        </div>

        <div id="chat-end"></div>
      </div>

      <div class="chat-input-bar">
        <input
          v-model="query"
          type="text"
          placeholder="继续提问..."
          class="chat-input"
          @keyup.enter="ask()"
          :disabled="loading"
        />
        <button class="send-btn" @click="ask()" :disabled="loading || !query.trim()">
          {{ loading ? '分析中...' : '发送' }}
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.app { min-height: 100vh; }

/* ── Home ── */
.home { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 70vh; padding: 0 1rem; }
.spacer { height: 6vh; }
.logo { width: 100px; margin-bottom: 1rem; border-radius: 50%; }
.title { font-size: 2.2rem; font-weight: 700; color: var(--text); margin-bottom: 0.3rem; }
.tagline { font-size: 1.05rem; color: var(--text-secondary); margin-bottom: 2rem; }
.search-box { display: flex; width: 100%; max-width: 600px; gap: 0.5rem; }
.search-input { flex: 1; padding: 0.85rem 1.2rem; font-size: 1rem; border: 2px solid var(--border); border-radius: var(--radius); outline: none; transition: border-color 0.2s; }
.search-input:focus { border-color: var(--primary); }
.search-btn { padding: 0.85rem 1.5rem; font-size: 1rem; background: var(--primary); color: #fff; border: none; border-radius: var(--radius); cursor: pointer; font-weight: 600; white-space: nowrap; }
.search-btn:hover { background: var(--primary-dark); }
.search-btn:disabled { opacity: 0.5; cursor: not-allowed; }

.examples { margin-top: 2rem; width: 100%; max-width: 600px; }
.examples-label { font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.6rem; }
.example-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.4rem; }
.example-btn { text-align: left; padding: 0.6rem 0.9rem; font-size: 0.82rem; background: var(--white); border: 1px solid var(--border); border-radius: 10px; cursor: pointer; color: var(--text); transition: all 0.15s; line-height: 1.3; }
.example-btn:hover { border-color: var(--primary); background: var(--primary-light); color: var(--primary); }

/* ── Chat ── */
.chat { max-width: 800px; margin: 0 auto; padding: 1rem; min-height: 100vh; display: flex; flex-direction: column; height: 100vh; }
.chat-header { display: flex; justify-content: space-between; align-items: center; padding: 0.6rem 0; border-bottom: 1px solid var(--border); margin-bottom: 0.5rem; flex-shrink: 0; }
.chat-title { font-size: 1.05rem; font-weight: 700; }
.new-chat-btn { padding: 0.35rem 0.85rem; font-size: 0.82rem; background: none; border: 1px solid var(--border); border-radius: 8px; cursor: pointer; color: var(--text-secondary); }
.new-chat-btn:hover { background: var(--primary-light); color: var(--primary); }

.chat-messages { flex: 1; overflow-y: auto; padding-bottom: 1rem; }
.msg { margin-bottom: 1.2rem; }
.msg-label { font-size: 0.78rem; color: var(--text-secondary); margin-bottom: 0.3rem; font-weight: 500; }
.msg.user .msg-label { text-align: right; }
.msg-content { line-height: 1.7; font-size: 0.95rem; }
.msg.user .msg-content { background: var(--primary-light); color: var(--primary-dark); margin-left: auto; max-width: 80%; padding: 0.8rem 1rem; border-radius: var(--radius); border-bottom-right-radius: 4px; text-align: left; }
.msg.assistant .msg-content { background: var(--white); border: 1px solid var(--border); max-width: 100%; padding: 1rem 1.5rem; border-radius: var(--radius); border-bottom-left-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }

.msg-content :deep(h1) { font-size: 1.4rem; margin: 0.8rem 0 0.4rem; }
.msg-content :deep(h2) { font-size: 1.2rem; margin: 0.7rem 0 0.35rem; }
.msg-content :deep(h3) { font-size: 1.08rem; margin: 0.5rem 0 0.25rem; }
.msg-content :deep(p) { margin: 0.4rem 0; }
.msg-content :deep(ul), .msg-content :deep(ol) { padding-left: 1.4rem; margin: 0.4rem 0; }
.msg-content :deep(li) { margin: 0.2rem 0; }
.msg-content :deep(code) { background: #F1F5F9; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.88em; }
.msg-content :deep(pre) { background: #1E293B; color: #E2E8F0; padding: 0.8rem; border-radius: 8px; overflow-x: auto; margin: 0.6rem 0; }
.msg-content :deep(pre code) { background: none; padding: 0; color: inherit; }
.msg-content :deep(table) { border-collapse: collapse; width: 100%; margin: 0.6rem 0; font-size: 0.9rem; }
.msg-content :deep(th), .msg-content :deep(td) { border: 1px solid var(--border); padding: 0.4rem 0.6rem; text-align: left; }
.msg-content :deep(th) { background: #F8FAFC; font-weight: 600; }
.msg-content :deep(blockquote) { border-left: 3px solid var(--primary); padding: 0.25rem 0.8rem; margin: 0.6rem 0; background: var(--primary-light); border-radius: 0 8px 8px 0; }
.msg-content :deep(strong) { font-weight: 700; }
.msg-content :deep(hr) { border: none; border-top: 1px solid var(--border); margin: 0.8rem 0; }

/* Metrics */
.metrics { display: flex; gap: 0.5rem; flex-wrap: wrap; font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid var(--border); }
.metric { background: var(--primary-light); padding: 0.15rem 0.5rem; border-radius: 6px; }
.metric.web { background: #DCFCE7; color: #166534; }

/* Feedback */
.feedback-row { display: flex; gap: 0.3rem; margin-top: 0.4rem; }
.fb-btn { background: none; border: 1px solid var(--border); border-radius: 6px; padding: 0.2rem 0.5rem; cursor: pointer; font-size: 0.9rem; line-height: 1; }
.fb-btn:hover { background: var(--primary-light); border-color: var(--primary); }

/* Sources */
.sources { margin-top: 0.5rem; font-size: 0.85rem; }
.sources summary { cursor: pointer; color: var(--primary); font-weight: 500; padding: 0.3rem 0; font-size: 0.82rem; }
.source-item { padding: 0.4rem 0; border-bottom: 1px solid var(--border); font-size: 0.82rem; }
.source-item:last-child { border-bottom: none; }
.source-score { color: var(--text-secondary); font-size: 0.78rem; margin-left: 0.3rem; }
.source-type { color: #166534; font-size: 0.75rem; margin-left: 0.3rem; background: #DCFCE7; padding: 0.05rem 0.35rem; border-radius: 4px; }
.source-snippet { color: var(--text-secondary); font-size: 0.8rem; margin: 0.15rem 0 0; font-style: italic; }

/* Streaming placeholder — shown inline while waiting for first token */
.streaming-placeholder { display: flex; align-items: center; gap: 0.6rem; padding: 0.8rem 0; color: var(--text-secondary); font-size: 0.9rem; }
.streaming-text { animation: fadeInOut 1.5s ease-in-out infinite; }
@keyframes fadeInOut { 0%, 100% { opacity: 0.5; } 50% { opacity: 1; } }

.dot-pulse { display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: var(--primary); animation: pulse 1.2s ease-in-out infinite; flex-shrink: 0; }
@keyframes pulse { 0%, 100% { opacity: 0.3; transform: scale(0.8); } 50% { opacity: 1; transform: scale(1.2); } }

/* Input bar */
.chat-input-bar { display: flex; gap: 0.5rem; padding: 0.8rem 0; border-top: 1px solid var(--border); background: rgba(255,255,255,0.9); backdrop-filter: blur(10px); flex-shrink: 0; }
.chat-input { flex: 1; padding: 0.75rem 1rem; font-size: 0.95rem; border: 2px solid var(--border); border-radius: var(--radius); outline: none; }
.chat-input:focus { border-color: var(--primary); }
.send-btn { padding: 0.75rem 1.2rem; background: var(--primary); color: #fff; border: none; border-radius: var(--radius); cursor: pointer; font-weight: 600; white-space: nowrap; }
.send-btn:hover { background: var(--primary-dark); }
.send-btn:disabled { opacity: 0.5; cursor: not-allowed; }

/* ── Identity ── */
.identity-bar-home { display: flex; align-items: center; gap: 0.4rem; margin: 1rem 0 0.5rem; flex-wrap: wrap; justify-content: center; }
.identity-label { font-size: 0.82rem; color: var(--text-secondary); white-space: nowrap; }
.identity-chip { padding: 0.3rem 0.7rem; font-size: 0.8rem; border: 1px solid var(--border); border-radius: 20px; background: var(--white); cursor: pointer; color: var(--text-secondary); transition: all 0.15s; }
.identity-chip:hover { border-color: var(--primary); color: var(--primary); }
.identity-chip.active { background: var(--primary); color: #fff; border-color: var(--primary); }

.header-right { display: flex; align-items: center; gap: 0.5rem; }
.identity-selector-compact { position: relative; }
.identity-toggle { padding: 0.35rem 0.7rem; font-size: 0.8rem; border: 1px solid var(--border); border-radius: 8px; background: none; cursor: pointer; color: var(--text-secondary); white-space: nowrap; }
.identity-toggle:hover { border-color: var(--primary); color: var(--primary); }
.identity-dropdown { position: absolute; right: 0; top: 100%; margin-top: 4px; background: var(--white); border: 1px solid var(--border); border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); z-index: 10; min-width: 160px; overflow: hidden; }
.id-option { display: flex; flex-direction: column; align-items: flex-start; width: 100%; padding: 0.5rem 0.8rem; border: none; background: none; cursor: pointer; text-align: left; border-bottom: 1px solid var(--border); font-size: 0.82rem; }
.id-option:last-child { border-bottom: none; }
.id-option:hover { background: var(--primary-light); }
.id-option.active { background: var(--primary-light); color: var(--primary); }
.id-label { font-weight: 500; }
.id-desc { font-size: 0.75rem; color: var(--text-secondary); }

@media (max-width: 768px) {
  .title { font-size: 1.7rem; }
  .search-box { flex-direction: column; }
  .search-btn { width: 100%; }
  .example-grid { grid-template-columns: 1fr; }
  .chat { padding: 0.5rem; height: 100vh; }
  .msg.user .msg-content { max-width: 90%; }
  .msg.assistant .msg-content { padding: 0.7rem 0.9rem; }
}
</style>
