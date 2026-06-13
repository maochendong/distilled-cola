# Plan — 蒸馏小可乐

> Architecture Design for SpecKit Workflow
> Phase: Plan v2.0 (updated 2026-06-11)

---

## 0. Current Production Architecture (from README + Source)

### 实际源码结构 (2026-06-11)
```
src/
├── config.py              # 主配置 (DeepSeek/ChromaDB/路径)
├── llm.py                 # LLM 客户端
├── collector/             # 视频采集: Whisper转录 + OCR + 标注
├── knowledge_base/        # 向量存储 + 混合检索 + 精排 + 推理链索引
│   ├── vector_store.py    # ChromaDB wrapper
│   ├── embedder.py        # bge-m3 / OpenAI embedding
│   ├── hybrid_retriever.py # BM25 + 语义 + RRF
│   ├── reranker.py        # Cross-encoder
│   └── reasoning_index.py # 推理链存储
├── rag/                   # RAG 管线
│   ├── pipeline.py        # ask() 编排 (v2.0: +streaming +conversation)
│   ├── retriever.py       # 检索器封装
│   ├── generator.py       # DeepSeek 生成 (v2.0: +streaming)
│   ├── reasoner.py        # 教师验证 + 溯源
│   ├── web_search.py      # AnySearch 实时搜索
│   ├── validator.py       # [NEW] 指数退避自检
│   ├── conversation.py    # [NEW] 多轮对话管理
│   └── reranker.py        # [NEW] 非阻塞精排
├── app/                   # 交互层
│   ├── cli.py             # Typer CLI (cola ask/import/serve)
│   ├── api.py             # FastAPI /ask, /stats, /health
│   ├── ui.py              # Streamlit UI (v3.0: 蓝白主题+对话+对比)
│   ├── feedback.py        # [NEW] 用户反馈 JSONL
│   └── comparison.py      # [NEW] 板块对比引擎
└── data/
    ├── raw/               # 原始转录
    ├── processed/         # 标注知识块
    ├── ocr_results/       # OCR 文本
    └── embeddings/        # ChromaDB 持久化
```

### 关键集成点 (后续迭代不能破坏)
1. **两路数据**: OCR + 转录 → ChromaDB `sh_knowledge` 单集合
2. **AnySearch**: domain=home, zone=cn, 自动补"上海"前缀
3. **教师验证**: DeepSeek Pro 6项检查 → confidence < 0.6 重试
4. **CLI**: `cola ask/import-video/serve/stats/build/export`
5. **API**: FastAPI /ask (JSON) + /stats + /health
6. **Docker**: Nginx:80 → Streamlit:8501, HF 模型缓存持久化

---

## 1. System Architecture Overview (Updated)

```
┌─────────────────────────────────────────────────────────────────┐
│                        User (Browser/Mobile)                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTPS
┌───────────────────────────▼─────────────────────────────────────┐
│                      Nginx (Reverse Proxy)                        │
│                  80 → 8501 · WebSocket · Gzip                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                   Streamlit UI (Vue/React SPA)                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────┐  │
│  │ Identity │  │ Deep     │  │ Scorer   │  │ AI Advisor     │  │
│  │ Selector │  │ Question-│  │ Display  │  │ (NL→Top N      │  │
│  │ (#40/69) │  │ naire    │  │ Cards    │  │  Recommendation)│  │
│  └──────────┘  └──────────┘  └──────────┘  └────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    RAG Pipeline (Python)                          │
│                                                                  │
│  ┌─────────────────────┐    ┌────────────────────────────────┐  │
│  │  ProfileRouter #53  │───▶│  Prompt Template Pool #52      │  │
│  │  (Identity → Route) │    │  FirstHome / HerHome /         │  │
│  └─────────────────────┘    │  FamilyUpgrade / GoldenYears   │  │
│                              └──────────┬─────────────────────┘  │
│                                         │                        │
│  ┌──────────────────────────────────────▼──────────────────────┐ │
│  │           Hybrid Retriever                                   │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │ │
│  │  │ Semantic     │  │ BM25         │  │ Cross-encoder    │   │ │
│  │  │ (bge-m3 +    │  │ (jieba +     │  │ Reranker         │   │ │
│  │  │  ChromaDB)   │  │  rank_bm25)  │  │ (bge-reranker)   │   │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘   │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                         │                        │
│  ┌──────────────────────────────────────▼──────────────────────┐ │
│  │           Scorer Framework #54                               │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐ │ │
│  │  │ Commute  │ │ Safety   │ │SchoolDist│ │ ElderlyFitness │ │ │
│  │  │ Scorer   │ │ Scorer   │ │ Scorer   │ │ Scorer         │ │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └────────────────┘ │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                         │                        │
│  ┌──────────────────────────────────────▼──────────────────────┐ │
│  │  Generator (DeepSeek V4) + Teacher-Student Validation        │ │
│  │  + Composite Scoring Engine #68                              │ │
│  └──────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                     Data Layer                                    │
│                                                                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐ │
│  │ ChromaDB   │  │ BM25 Index │  │ Structured │  │ Video      │ │
│  │ (Vectors)  │  │ (Keywords) │  │ DB (SQLite)│  │ Knowledge  │ │
│  └────────────┘  └────────────┘  └────────────┘  │ Pipeline   │ │
│                                                    └────────────┘ │
│  Knowledge Collections: SH / FirstHome / HerHome / Family / Golden │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Module Architecture

### 2.1 Identity & Routing Layer

```
┌─ User Entry ──────────────────────────────────────────────┐
│  [Identity Card Selection] → [Deep Questionnaire #69]      │
│       → Profile Storage #40 → Next visit auto-recognized  │
└───────────────────────────────────────────────────────────┘
                              │
┌─ ProfileRouter #53 ────────▼─────────────────────────────┐
│  match_identity(profile) → route_to_prompt_template()      │
│                           → route_to_scorer_combo()         │
│                           → adjust_retrieval_weights()      │
└───────────────────────────────────────────────────────────┘
```

**Scorer组合映射：**

| Identity | Prompt Theme | Active Scorers |
|----------|-------------|----------------|
| FirstHome | 通勤代价 × 上车门槛 | CommuteScorer + AppreciationScorer |
| HerHome | 安全底线 × 产权规划 | SafetyScorer + LiquidityScorer + PrivacyScorer |
| FamilyUpgrade | 学区逻辑 × 置换链 | SchoolDistrictScorer + FamilyScorer |
| GoldenYears | 医疗保障 × 无障碍 | ElderlyFitnessScorer + MedicalScorer |
| General | 板块价值综合分析 | (none, general RAG) |

### 2.2 Data Layer Architecture

```
data/
├── embeddings/
│   ├── sh_knowledge/          # 上海房产通用知识 (现有)
│   ├── firsthome/             # [Phase 3] 首套刚需
│   ├── herhome/               # [Phase 3] 女性购房
│   ├── family_upgrade/        # [Phase 3] 家庭改善
│   └── golden_years/          # [Phase 3] 养老置业
├── metadata/
│   ├── commute_scores.csv     # [Phase 3] 通勤数据
│   ├── safety_scores.csv      # [Phase 3] 安全数据
│   ├── school_districts.db    # [Phase 3] 学区数据
│   ├── elderly_poi.db         # [Phase 3] 养老POI
│   └── property_ratings.json  # [Phase 3] 物业评级
└── chroma_db/                 # ChromaDB persistent storage
```

---

## 3. Implementation Phases (Task Execution Order)

### Phase 1 — MVP 基座 (Week 1-2)

```
Priority: CRITICAL
Dependency: None (start here)

T-001 (P0) 多轮对话       ─┐
T-002 (P0) 流式输出       ─┤  → 核心交互体验
T-003 (P1) 用户反馈       ─┘
T-004 (P1) 自检优化       ─┐
T-005 (P1) 异常告警       ─┤  → 系统健壮性
T-006 (P2) 缓存日志       ─┘
```

### Phase 2 — 服务差异化启动 (Month 1)

```
Priority: HIGH
Dependency: Phase 1 complete

┌── 交互升级 ──────────────────────────────┐
│ T-007 (P0) Web UI 重构 ← 必须先做        │
│ T-008 (P0) 移动端适配                    │
│ T-009 (P0) 板块对比模块                  │
│ T-010 (P1) 数据可视化                    │
│ T-011 (P1) 多模式回答                    │
│ T-021 (P2) 智能搜索联想                  │
└──────────────────────────────────────────┘

┌── 差异化核心 ← 关键路径 ────────────────┐
│ T-015 (P1) 用户画像采集 + 身份路由       │ → 所有差异化的前提
│ T-016 (P1) 需求深度问卷                   │ → 画像的数据输入
│ T-012 (P1) 房贷计算器                    │
│ T-013 (P1) 板块评分系统                  │
│ T-014 (P1) 问答缓存优化                  │
└──────────────────────────────────────────┘

┌── 知识注入 ──────────────────────────────┐
│ T-017 (P1) [FirstHome] 知识注入          │
│ T-018 (P1) [FamilyUpgrade] 知识注入      │
│ T-019 (P2) [HerHome] 知识注入            │
│ T-020 (P2) 高级搜索筛选器                │
└──────────────────────────────────────────┘
```

### Phase 3 — 数据底座 (Month 2-3)

```
Priority: HIGH
Dependency: Phase 2 completed

Parallel work streams:

Stream A: 通用数据                    Stream B: 垂直数据
┌─────────────────────────────┐     ┌─────────────────────────┐
│ T-022 楼盘级数据库           │     │ T-027 [FirstHome] 通勤   │
│ T-023 土地市场数据           │     │ T-028 [Family] 学区/儿童 │
│ T-024 学区数据库             │     │ T-029 [HerHome] 安全     │
│ T-025 地铁/规划数据          │     │ T-030 [Golden] 生活圈POI │
│ T-026 多博主蒸馏（1→3-5）   │     │ T-031 [Golden] 物业评级   │
│ T-032 宏观经济指标（P2）     │     │ T-033 [Golden] 养老设施库 │
│ T-034 知识盲区检测（P2）     │     └─────────────────────────┘
└─────────────────────────────┘
```

### Phase 4 — 差异化能力上线 (Month 3-4)

```
Priority: HIGH
Dependency: Phase 3 data ready

┌── FirstHome ─────┐  ┌── HerHome ──────────┐
│ T-038 通勤增值评分│  │ T-040 安全评分卡     │
│ T-039 上车决策    │  │ T-041 贷款/产权规划  │
└──────────────────┘  │ T-070 独居适配度报告  │
                       │ T-046 流动性评估(P2)  │
┌── FamilyUpgrade ──┐  └─────────────────────┘
│ T-042 学区社区评分 │
│ T-043 置换链条规划 │  ┌── GoldenYears ──────┐
└──────────────────┘  │ T-044 适老评估        │
                       │ T-045 子女协作        │
┌── 通用 ───────────┐  │ T-047 置换规划(P2)   │
│ T-035 价格异动提醒 │  └─────────────────────┘
│ T-036 关注收藏     │
│ T-037 多模态搜索   │  ┌── P2 补充 ─────────┐
│ T-048 专家咨询(P2) │  │ T-049 趋势预测      │
│ T-050 语音输入(P2) │  │ T-038 循环...       │
└──────────────────┘  └─────────────────────┘
```

### Phase 5 — 基础设施 (Ongoing from Month 2)

```
Can start in parallel with Phase 3-4:

┌── 路由与模板 ─────────────────────────────┐
│ T-053 (P1) 多身份 Prompt 模板系统          │
│ T-054 (P1) ProfileRouter 路由模块           │
└────────────────────────────────────────────┘

┌── 工程化 ──────────────────────────────────┐
│ T-051 (P1) A/B 测试框架                    │
│ T-052 (P1) 监控与日志体系                  │
│ T-055 (P2) 知识库增量更新                  │
│ T-056 (P2) 知识库版本管理                  │
│ T-057 (P2) Scorer 框架抽象                 │
│ T-058 (P2) 综合评分引擎                    │
│ T-059 (P2) 国际化                          │
└────────────────────────────────────────────┘
```

### Phase 6 — 商业化 (On-demand)

```
No hard dependency. Start when C-side value is validated.

T-060 付费报告 → T-062 会员体系 → T-063 B2B API
   ↓              ↓
T-061 新盘导流    T-064~067 B2B 赋能工具
   ↓
T-071 垂直人群社区 + T-072 服务商生态
   ↓
T-068 + T-033 内容矩阵
```

---

## 4. Dependency Graph (Simplified)

```
[T-001 多轮对话] ─┐
[T-002 流式输出] ─┤
[T-003 用户反馈] ─┤
[T-004 自检优化] ─┤ → Phase 2
[T-005 异常告警] ─┘
[T-006 缓存日志] ─┘

[T-007 Web UI]    ─┐
[T-015 画像路由]  ─┤ → Phase 3 (数据生产)
[T-016 需求问卷]  ─┘
[T-017~019 注入]  ─┘

[T-022~031 数据] ─┐ → Phase 4 (评分器消费)
[T-053 Prompt]   ─┤
[T-054 Router]   ─┘

[T-038~045 评分器] → T-058 综合评分引擎
                  → Phase 6 商业化
```

---

## 5. Key Architecture Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | **Identity-first routing** over monolithic pipeline | 前期成本高，但每个新身份模块只需加 Prompt + Scorer + Data，不碰核心管线 |
| 2 | **Scorer plugin architecture** | SafetyScorer / CommuteScorer 等可独立开发测试，通过统一接口注入答案上下文 |
| 3 | **复用现有视频蒸馏管道** | 新知识线不重复造轮子，通过现有 Whisper+OCR→ChromaDB 流程即可注入 |
| 4 | **Prompt as configuration** | 每个身份模块的 prompt 模板独立成文件，不硬编码在代码中，可动态加载和 AB 测试 |
| 5 | **Phase 6 最后启动** | C 端价值跑通前不做商业化，避免过早优化收入模型 |
