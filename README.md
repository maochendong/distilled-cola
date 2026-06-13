# 蒸馏小可乐 🏠

> **助你决策人生最大一笔投资** — 将房产博主的专业知识「蒸馏」为可交互的 AI 分析系统，融合实时行情数据

把一位深耕上海房产市场的视频博主的**分析框架、推理路径、决策逻辑**完整蒸馏出来，结合 **AnySearch 实时搜索引擎**获取最新成交数据、挂牌信息和政策动态，同时接入 **SQLite 结构化楼盘数据库**，让你随时能问出「前滩均价多少？」「800 万预算前滩 vs 大宁怎么选？」这类问题，得到一个像资深分析师一样的专业回答。

---

## 目录

- [核心思路](#核心思路)
- [系统架构](#系统架构)
- [数据流](#数据流)
- [快速开始](#快速开始)
- [视频下载](#视频下载)
- [画面文字提取 (OCR)](#画面文字提取-ocr)
- [可用命令](#可用命令)
- [项目结构](#项目结构)
- [技术栈](#技术栈)

---

## 核心思路

### 知识蒸馏，而非模型训练

目标不是训练一个新模型，而是构建一套 **结构化知识检索 + 思维链增强生成** 系统：

```
博主视频 → 音频转录 / 画面OCR（独立流水线）
     → 各自切分 → 各自标注 → 各自嵌入 → 同一 ChromaDB
     → 混合检索 + 四步分析 RAG → 专家级回答
```

两路数据**分开处理、同库存储**，每块带 `source` 标签（`ocr` / `transcription`），检索时双路并查：

| 数据源 | 处理入口 | 内容特点 |
|--------|----------|----------|
| **画面 OCR** | `ingest_all.py` | 图表数据、成交价、挂牌量、政策原文 |
| **音频转录** | `ingest_transcriptions.py` | 博主口播分析、推理逻辑、建议 |

### 三层提示词架构

生成回答时使用三层结构化提示：

| 层级 | 内容 |
|------|------|
| **身份层** | 博主的数字分身，拥有一致的分析框架和表达风格 |
| **分析流程层** | 强制四步推演：识别变量 → 调用历史框架 → 代入数据 → 给出建议 |
| **约束层** | 强制溯源、置信度评分、风险声明 |

### 四步分析流程

每次问答严格遵循：

1. **识别关键变量** — 板块定位、供需关系、政策导向、价格水平、流动性、用户画像
2. **调用历史框架** — 检索博主在类似问题上的分析路径和推理链
3. **代入当前数据** — 板块最新成交价、挂牌量、政策环境代入推演
4. **给出个性化建议** — 按短期/中期/长期 + 风险提示的结构输出

---

## 系统架构

```
                           ┌─────────────────────┐
                           │   数据源（多种输入）   │
                           │ 微信视频号/小红书/YouTube│
                           │ 链家/贝壳 CSV / 爬虫  │
                           └──────────┬──────────┘
                                      │
                     ┌────────────────┼────────────────────┐
                     │                │                    │
          ┌──────────▼──────────┐  ┌─▼────────────────┐  ┌─▼──────────────┐
          │  音频转录 (Whisper)  │  │ 画面 OCR         │  │ 结构化数据      │
          │  230 mp4 → 转写文本  │  │ OpenCV+PaddleOCR │  │ SQLite 8表     │
          └──────────┬──────────┘  └──┬───────────────┘  │ 楼盘/挂牌/成交  │
                     │                │                  │ 学区/地铁/学校  │
                     └──────┬─────────┘                  └──────┬─────────┘
                            ▼                                  │
                 ┌─────────────────────┐                       │
                 │  ChromaDB + BM25     │◄──────────────────────┘
                 │  sh_knowledge        │    四路混合检索
                 │  (ocr + transcription)│
                 └──────────┬──────────┘
                            ▼
                 ┌──────────────────────────────────────────────┐
                 │  四路混合检索（HybridRetriever）               │
                 │  ┌──────────┐ ┌──────────┐ ┌──────────┐     │
                 │  │ 语义向量  │ │ BM25 关键词│ │ 推理链   │     │
                 │  │ ChromaDB │ │ rank-bm25 │ │ 排序加分 │     │
                 │  └──────────┘ └──────────┘ └──────────┘     │
                 │  ┌──────────┐ + Cross-encoder 精排           │
                 │  │结构化数据 │ ← SQLite 板块/房源实时查询       │
                 │  └──────────┘                                │
                 └──────────────────────────────────────────────┘
                            │
                            ▼
                 ┌──────────────────────────────────────┐
                 │     AnySearch 实时搜索引擎              │
                 │     domain=home, zone=cn, 上海限定     │
                 │     freshness=year, 结果数 5           │
                 └──────────────────────────────────────┘
                            │
                            ▼
                 ┌──────────────────────────────────────┐
                 │   DeepSeek 四步分析生成 + 教师自检      │
                 │   + 身份感知 Prompt（5 种身份模板）      │
                 └──────────────────────────────────────┘
                            │
                            ▼
                 ┌──────────────────────────────────────┐
                 │   交互层 (三种前端)                     │
                 │   Vue SPA / Streamlit / Typer CLI      │
                 │   多轮对话 · 流式输出 · 身份选择        │
                 │   板块对比 · 用户反馈 · 纠错输入        │
                 └──────────────────────────────────────┘
```

---

## 数据流

### 视频处理流程

```
视频文件 (mp4)
    │
    ├──→ 自动检测视频类型 ──────────────────────────┐
    │      │                                        │
    │      ├──→ 有字幕流 → ffmpeg 直接提取字幕       │
    │      │              (最快，零 GPU，1 秒完成)    │
    │      │                                        │
    │      └──→ 无字幕流 → Whisper 音频转录          │
    │              │                                │
    │              ├──→ 口播正常 → 保留转录结果       │
    │              └──→ 纯音乐/无语音 → 丢弃转录      │
    │                                                │
    ├──→ OpenCV 帧提取（每 2 秒 + 场景变化检测）      │
    │       │                                        │
    │       └──→ PaddleOCR 画面文字识别               │
    │               │                                │
    │               └──→ GPT-4o Vision（复杂图表结构化分析）│
    │                                                │
    └──→ 合并文本：口播/字幕 + 画面文字 → 完整知识    │
            │                                        │
            └──→ 逻辑段落切分（检测话题转移信号）
                    │
                    └──→ 房产领域双维度标注（教师模型）
                            │
                            └──→ 入库：知识索引 + BM25 倒排索引
```

### 问答流程

```
用户问题："前滩均价多少？" / "800万预算前滩vs大宁怎么选？"
    │
    ├──→ 关键变量提取（板块/预算/政策/供需/流动性/画像）
    │
    ├──→ 四路混合检索
    │       │
    │       ├──→ 语义向量检索 × BM25 关键词检索
    │       │       │
    │       │       └──→ RRF 融合排序 → Top-20 候选池
    │       │
    │       ├──→ 实体感知推理链检索
    │       │       │
    │       │       ├──→ 完整子串匹配板块名（最长优先，避免短名误匹配长名）
    │       │       ├──→ 同板块自动扩展（"前滩"→"前滩南""前滩九宫格"）
    │       │       ├──→ 关键字映射匹配逻辑标签（"对比"→板块对比，"学区"→学区分析等）
    │       │       └──→ 匹配板块 +0.15 / 匹配逻辑标签 +0.10 排序加分
    │       │
    │       ├──→ 结构化数据查询（SQLite）
    │       │       │
    │       │       ├──→ 板块均价/挂牌量/价格区间
    │       │       ├──→ 在售房源列表（户型/面积/总价）
    │       │       └──→ 学区/地铁/成交历史
    │       │
    │       ├──→ Cross-encoder 精排（BGE Reranker v2）→ Top-5
    │       │
    │       └──→ RAGPipeline 实例缓存（首次构建后复用）
    │
    ├──→ 身份感知（可选，5 种身份模板）
    │       │
    │       ├──→ 首套刚需：侧重通勤/总价门槛/首付月供
    │       ├──→ 独立女性：侧重安全/产权/流动性
    │       ├──→ 家庭改善：侧重学区/置换/儿童友好
    │       ├──→ 养老置业：侧重医疗/无障碍/生活圈
    │       └──→ 通用模式：标准四步分析
    │
    ├──→ 实时网络搜索（关键词自动触发）
    │       │
    │       ├──→ 查询自动追加"上海"前缀
    │       ├──→ domain=home, zone=cn, freshness=year
    │       └──→ 领域搜索无结果时降级通用搜索
    │
    ├──→ 组装提示词：身份层 + 分析流程层 + 约束层 + 显式引用标注
    │       │
    │       └──→ 注入 few-shot 推理范例 + 结构化数据（板块实时行情）
    │
    ├──→ DeepSeek Flash 流式生成四步结构化回答（SSE 逐 token 输出）
    │
    ├──→ 自检循环：教师模型（DeepSeek Pro）校验置信度 + 溯源真实性
    │       │
    │       ├──→ 6 项检查：变量识别/推理链/风险/建议/来源标注/引用真实性
    │       └──→ 置信度 < 0.6 或 ≥2 项未通过则重生成
    │
    └──→ 输出：结构化分析 + 来源编号引用 + 置信度 + 来源列表
```

---

## 快速开始

### 环境要求

- Python 3.10+（mlx-whisper 需要 3.10+，否则用标准 Whisper）
- `DEEPSEEK_API_KEY`（主推理引擎）

### 安装

```bash
# 1. 克隆 / 进入项目
cd 蒸馏小可乐

# 2. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 3. 安装依赖
pip install -e .

# 4. 可选：安装 OCR 引擎（用于视频画面文字识别）
pip install -e ".[ocr]"

# 5. 可选：安装 Whisper（用于本地视频音频转录）
pip install -e ".[transcribe]"

# 6. 配置 API Key
cp .env.example .env
# 编辑 .env，至少填入 DEEPSEEK_API_KEY（主推理）+ OPENAI_API_KEY（可选，用于 embedding 和 vision）
```

### 配置 `.env`

```env
# DeepSeek（主：问答、标注、质检）
DEEPSEEK_API_KEY=sk-your-deepseek-key
DEEPSEEK_BASE_URL=https://api.deepseek.com

# OpenAI（备选：embedding + 图片文字识别）
OPENAI_API_KEY=sk-your-openai-key

# 学生模型 ≈ 在线问答 | 教师模型 ≈ 标注/知识蒸馏
STUDENT_MODEL=deepseek-v4-flash
TEACHER_MODEL=deepseek-v4-pro

# 嵌入模型（本地默认 bge-m3，设此值后改用 OpenAI）
# EMBEDDING_MODEL=text-embedding-3-small
```

---

## 视频下载

在导入视频前，需要先从目标平台批量下载博主的视频。以下是两个平台的推荐工具。

### 微信视频号

**工具**: [wx_channels_download](https://github.com/ltaoo/wx_channels_download)

通过本地 HTTPS 代理拦截微信 PC 客户端请求，在博主主页注入「批量下载」按钮，支持一键下载全部视频。

### 小红书

**工具**: [XHS-Downloader](https://github.com/JoeanAmier/XHS-Downloader)

Docker 一键部署，输入博主主页链接即可自动提取并下载所有图文和视频（支持无水印）。

---

## 画面文字提取 (OCR)

对于纯音乐+文字类的视频，使用 OCR 从视频帧中提取画面文字。

### 方案

| 方案 | 引擎 | 适用场景 |
|------|------|----------|
| **macOS Vision 框架** | 系统原生 | 批量处理图片帧，中文识别优秀，零依赖无需下载模型 |
| **PaddleOCR** | 离线 | 高精度中文识别（需安装 `pip install -e ".[ocr]"`） |
| **GPT-4o Vision** | API | 复杂图表/结构化提取，PaddleOCR 失败时回退 |

### 批量处理图片帧

下载的视频解帧后，每个子文件夹对应一个独立问答主题，图片按数字顺序排列叙事。

```bash
# OCR 全部子文件夹中的图片
python scripts/batch_ocr_video.py

# 清理水印、OCR 伪影等干扰文字
python scripts/clean_ocr.py
```

输出保存在 `data/ocr_results/`：

```
  _summary.json       # 全部主题汇总（309 个视频，45 万字）
  主题A.txt            # 每个主题完整转录，按图片顺序拼接
  主题B.txt
  ...
```

### 音频转录（mlx-whisper）

230 个原始 mp4 使用 Apple Silicon 优化的 mlx-whisper 批量转录：

```bash
# 批量转录（自动跳过已完成，支持断点续传）
python scripts/batch_transcribe_mlx.py
```

转录结果保存到 `data/raw/`，与 OCR 文本融合后用于蒸馏入库。

### 蒸馏入库

两路数据**分开处理、同库存储**，每块知识带 `source` 标签标注来源：

```bash
# 1. OCR 数据入库（清空旧索引后全量重建）
python scripts/ingest_all.py

# 2. 转录数据追加入库（增量追加到同一 ChromaDB）
python scripts/ingest_transcriptions.py
```

OCR 数据（画面图表、成交数据）和处理转录数据（博主口播分析）独立切分与标注，最终汇聚到同一知识索引，问答时混合检索两路结果。

---

## 可用命令

### 数据导入

```bash
# 导入本地视频（微信视频号/小红书下载的 mp4）
cola import-video ~/Downloads/前滩分析.mp4 --platform local --title "前滩板块深度分析"
# 启用 OCR 自动提取画面文字（默认开启），可关闭：
cola import-video 视频.mp4 --platform local --no-ocr

# 导入单张图片（数据截图/政策文件/户型图）
cola import-image 上海新政截图.png --title "2024年上海楼市新政"

# 从 YouTube 导入
cola import-video dQw4w9WgXcQ --title "上海楼市分析"
```

### 问答

```bash
# CLI 问答
cola ask "800万预算，前滩vs大宁怎么选？"
cola ask "现在是上海买房的好时机吗？" --top-k 8

# 启动 Web 界面
cola streamlit

# 启动 API 服务
cola serve
```

### 知识库管理

```bash
# 查看知识库统计
cola stats

# 手动构建知识库（从已有的标注数据）
cola build --input data/processed/annotated_blocks.json

# 导出知识库
cola export knowledge_export.json
```

### 结构化数据导入

```bash
# 从 CSV 导入楼盘/房源数据（支持板块/楼盘/户型/价格）
cola data-import data/metadata/sample_shanghai.csv

# 先生成示例数据再导入（4 板块、10 楼盘、12 套房源）
cola data-import data/metadata/sample_shanghai.csv --sample

# 查看结构化数据统计
cola data-stats
```

### 典型问答场景

| 场景 | 示例问题 |
|------|----------|
| 行情查询 | `前滩目前均价多少？` |
| 板块选择 | `总价500万，上海哪里性价比最高？` |
| 预算匹配 | `300万预算首套上车，通勤和增值哪个更重要？` |
| 买卖决策 | `现在该卖房吗，还是再等等？` |
| 新房/二手 | `上海现在买房，选新房还是二手次新？` |
| 价格判断 | `看中一套房，怎么判断价格合不合理？` |

---

## 项目结构

```
蒸馏小可乐/
├── src/
│   ├── collector/                # 数据采集与处理
│   │   ├── transcriber.py        # Whisper 音频转录 + 字幕流提取 + 说话人分离
│   │   ├── frame_extractor.py    # OpenCV 视频帧提取（智能采样）
│   │   ├── ocr.py                # 双引擎 OCR（PaddleOCR + Vision API）
│   │   ├── segmenter.py          # 逻辑段落切分（话题转移检测）
│   │   ├── annotator.py          # 房产领域双维度标注（实体/逻辑/建议）
│   │   └── pipeline.py           # 端到端采集管道（自动检测视频类型）
│   ├── knowledge_base/           # 知识库
│   │   ├── embedder.py           # 向量嵌入（OpenAI / 本地模型）
│   │   ├── vector_store.py       # 知识索引（ChromaDB，支持标签过滤）
│   │   ├── reasoning_index.py    # 推理链索引
│   │   ├── reranker.py           # Cross-encoder 精排（BGE Reranker v2）
│   │   └── hybrid_retriever.py   # 四路混合检索（语义+BM25+推理链+结构化数据）
│   ├── data/                     # 结构化数据层
│   │   ├── db.py                 # SQLite 数据库（8 表：楼盘/挂牌/成交/学区/地铁...）
│   │   ├── loaders/
│   │   │   ├── lianjia.py        # 链家/贝壳 CSV 导入 + 贝壳页面抓取
│   │   └── scorers/              # 板块评分器（Phase 4）
│   ├── rag/                      # RAG 管道
│   │   ├── generator.py          # 三层提示词架构生成器（支持流式）
│   │   ├── retriever.py          # 混合检索器封装
│   │   ├── pipeline.py           # 四步分析流程 + 自检循环 + 实时搜索
│   │   ├── conversation.py       # 多轮对话管理
│   │   ├── validator.py          # 教师模型自检（指数退避重试）
│   │   ├── web_search.py         # AnySearch 实时搜索引擎客户端
│   │   └── reasoning.py          # 回答质量验证 + 溯源格式化
│   ├── app/                      # 交互层
│   │   ├── cli.py                # Typer CLI（12 个子命令）
│   │   ├── api.py                # FastAPI 服务（/ask, /data/*, /feedback, /districts）
│   │   ├── ui.py                 # Streamlit 交互界面（身份选择 + 板块对比）
│   │   ├── feedback.py           # 用户反馈闭环
│   │   ├── comparison.py         # 板块对比引擎
│   │   └── static/               # Web UI 静态文件
│   └── config.py                 # 配置管理（环境变量 + 默认值）
├── frontend/                     # Vue SPA 前端
│   ├── src/App.vue               # 主组件（身份选择 + 流式对话 + 反馈）
│   └── vite.config.js            # Vite 构建配置
├── data/                         # 数据存储
│   ├── raw/                      # 原始转录/OCR 结果
│   ├── processed/                # 标注后的知识块
│   ├── ocr_results/              # 帧 OCR 文本
│   ├── metadata/                 # SQLite 结构化数据库
│   └── embeddings/               # ChromaDB 向量存储
├── scripts/                       # 实用脚本
├── tests/                        # 单元测试
├── .speckit/                     # 项目管理（71 项任务）
├── pyproject.toml
├── .env.example
└── README.md
```

---

## 技术栈

| 组件 | 方案 | 说明 |
|------|------|------|
| 视频类型检测 | **自动分流** | 字幕流 / 口播 / 纯音乐，自动选择最优路径 |
| 字幕流提取 | **ffmpeg** | 内嵌 SRT/ASS 字幕直接提取，零 GPU 消耗 |
| 音频转录 | **Whisper / mlx-whisper** | 标准版或 Apple Silicon 优化版，自动丢弃纯音乐视频 |
| 说话人分离 | **PyAnnote** (可选) | 区分博主独白 vs 采访对话 |
| 帧提取 | **OpenCV** | 每 2 秒采样 + 场景变化检测 |
| 文字识别 | **macOS Vision / PaddleOCR / GPT-4o Vision** | 三引擎，自动降级 |
| 文本嵌入 | **BGE-M3** (本地, 1024维) / **OpenAI** | 本地默认，OpenAI 备选 |
| 向量数据库 | **ChromaDB** | 本地持久化，支持元数据过滤 |
| 结构化数据 | **SQLite** | 楼盘/挂牌/成交/学区/地铁 8 张表，链家 CSV 导入 |
| 关键词检索 | **BM25 (rank-bm25)** | 确保精确查询命中 |
| 混合检索 | **四路 RRF 融合 + 实体感知排序 + 精排** | BM25 × 语义向量 × 推理链 × 结构化数据；Cross-encoder 二次精排 |
| 精排模型 | **BGE Reranker v2** | (query, doc) pair 重打分，候选池 ×4 扩召回后精排 |
| 置信度评估 | **教师模型验证** | DeepSeek Pro 做 6 项检查（含引用真实性核查）|
| 来源引用 | **显式编号标注** | 回答中标注 [1][2] 来源编号，validator 逐条核对真实性 |
| 推理 API | **DeepSeek v4 Flash** (学生) / **DeepSeek v4 Pro** (教师) | 主引擎，可替换 OpenAI |
| 实时搜索 | **AnySearch API** | domain=home 房产领域 + zone=cn 中国区域，上海限定，自动降级通用搜索 |
| 多轮对话 | **内存会话管理** | 自动创建 conv_id，上下文注入最近 5 轮对话 |
| 身份感知 | **5 套 Prompt 模板** | 首套刚需/独立女性/家庭改善/养老置业/通用 |
| 流式输出 | **SSE (Server-Sent Events)** | 逐 token 流式渲染，done 事件携带 sources |
| 前端 (Vue SPA) | **Vue 3 + Vite** | 身份选择 + 流式对话 + 反馈 + 板块对比 |
| 前端 (Streamlit) | **Streamlit** | 蓝白主题，身份选择，板块对比，纠错输入 |
| 交互界面 | **Typer CLI** + **FastAPI** + **Streamlit** | 四种交互方式 |

---

## 许可证

MIT
