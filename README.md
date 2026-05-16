# 蒸馏小可乐 🏠

> **上海房产分析专家** — 将房产博主的专业知识「蒸馏」为可交互的 AI 分析系统

把一位深耕上海房产市场的视频博主（微信视频号/小红书/YouTube）的**分析框架、推理路径、决策逻辑**完整蒸馏出来，让你随时能问出「800 万预算前滩 vs 大宁怎么选？」这类问题，得到一个像博主本人在微信语音里给你分析一样的专业回答。

---

## 目录

- [核心思路](#核心思路)
- [系统架构](#系统架构)
- [数据流](#数据流)
- [快速开始](#快速开始)
- [可用命令](#可用命令)
- [项目结构](#项目结构)
- [技术栈](#技术栈)

---

## 核心思路

### 知识蒸馏，而非模型训练

目标不是训练一个新模型，而是构建一套 **结构化知识检索 + 思维链增强生成** 系统：

```
博主视频 → 音频转录 + 画面OCR → 逻辑段落切分
     → 房产领域双维度标注 → 向量知识库 + 推理链索引
     → 四步分析 RAG → 专家级回答
```

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
                          └──────────┬──────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │        采集管道 (Collector)       │
                    │                                  │
                    │  ┌──────────┐  ┌──────────────┐  │
                    │  │Whisper   │  │OpenCV 帧提取  │  │
                    │  │音频转录  │  │+ PaddleOCR   │  │
                    │  │+说话人分离│  │+ Vision API  │  │
                    │  └────┬─────┘  └──────┬───────┘  │
                    │       └──────┬────────┘           │
                    │         ┌────▼─────┐              │
                    │         │逻辑段落切分│              │
                    │         └────┬─────┘              │
                    │         ┌────▼─────┐              │
                    │         │房产领域标注│              │
                    │         │(实体/逻辑/建议)          │
                    │         └──────────┘              │
                    └────────────────┬─────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │        知识库 (Knowledge Base)    │
                    │                                  │
                    │  ┌────────────┐ ┌──────────────┐ │
                    │  │ 知识索引    │ │ 推理链索引    │ │
                    │  │(ChromaDB)  │ │(ChromaDB)    │ │
                    │  └─────┬──────┘ └──────┬───────┘ │
                    │        └───────┬────────┘         │
                    │         ┌──────▼──────┐           │
                    │         │ 混合检索器   │           │
                    │         │ BM25 + 语义  │           │
                    │         │ + RRF 融合   │           │
                    │         └─────────────┘           │
                    └────────────────┬─────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │    RAG 管道 (四步分析)            │
                    │                                  │
                    │  ┌──────────┐ ┌──────────────┐  │
                    │  │ 混合检索  │ │ 三层提示词生成 │  │
                    │  │ + 推理链  │ │ + 自检循环   │  │
                    │  └──────────┘ └──────────────┘  │
                    └────────────────┬─────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │     交互层 (CLI / API / Streamlit)│
                    │                                  │
                    │   cola ask / cola serve / cola   │
                    │           streamlit              │
                    └──────────────────────────────────┘
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
用户问题："800万预算前滩vs大宁怎么选？"
    │
    ├──→ 关键变量提取（板块/预算/政策/供需/流动性/画像）
    │
    ├──→ 混合检索：语义向量检索 × BM25 关键词检索 × 推理链检索
    │       │
    │       └──→ RRF 融合排序 → Top-K 知识片段 + 推理链
    │
    ├──→ 组装提示词：身份层 + 分析流程层 + 约束层
    │       │
    │       └──→ 注入 few-shot 推理范例（博主历史分析路径）
    │
    ├──→ GPT-4o-mini 生成四步结构化回答
    │
    └──→ 自检循环：置信度 < 0.6 则重新生成
            │
            └──→ 输出：结构化分析 + 参考来源 + 置信度
```

---

## 快速开始

### 环境要求

- Python 3.9+
- `OPENAI_API_KEY`（GPT-4o-mini + text-embedding-3-small）

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

STUDENT_MODEL=deepseek-v4-flash
TEACHER_MODEL=deepseek-v4-pro
STUDENT_MODEL=gpt-4o-mini
TEACHER_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small
```

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

### 典型问答场景

| 场景 | 示例问题 |
|------|----------|
| 板块选择 | `800万预算，前滩vs大宁怎么选？` |
| 楼盘点评 | `云锦东方真的值得摇吗？` |
| 政策解读 | `上海认房不认贷后，置换链条怎么走？` |
| 时机判断 | `现在是上海买房的好时机吗？` |
| 学区分析 | `徐汇和浦东的学区房，哪个更抗跌？` |
| 新房/二手 | `2025年上海新房供应量对二手市场冲击多大？` |

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
│   │   └── hybrid_retriever.py   # BM25 + 语义混合检索（RRF 融合）
│   ├── rag/                      # RAG 管道
│   │   ├── generator.py          # 三层提示词架构生成器
│   │   ├── retriever.py          # 混合检索器封装
│   │   ├── pipeline.py           # 四步分析流程 + 自检循环
│   │   └── reasoning.py          # 回答质量验证 + 溯源格式化
│   ├── app/                      # 交互层
│   │   ├── cli.py                # Typer CLI（9 个子命令）
│   │   ├── api.py                # FastAPI 服务（/ask, /stats, /health）
│   │   └── ui.py                 # Streamlit 交互界面
│   └── config.py                 # 配置管理（环境变量 + 默认值）
├── data/                         # 数据存储
│   ├── raw/                      # 原始转录/OCR 结果
│   ├── processed/                # 标注后的知识块
│   └── embeddings/               # ChromaDB 向量存储
├── tests/                        # 单元测试
│   ├── test_segmenter.py         # 段落切分测试
│   ├── test_hybrid_retriever.py  # 混合检索测试
│   ├── test_ocr.py               # OCR + 帧提取测试
│   └── test_pipeline.py          # Pipeline 导入测试
├── notebooks/
│   └── analysis.ipynb            # Jupyter 探索 notebook
├── pyproject.toml
├── .env.example
└── .gitignore
```

---

## 技术栈

| 组件 | 方案 | 说明 |
|------|------|------|
| 视频类型检测 | **自动分流** | 字幕流 / 口播 / 纯音乐，自动选择最优路径 |
| 字幕流提取 | **ffmpeg** | 内嵌 SRT/ASS 字幕直接提取，零 GPU 消耗 |
| 音频转录 | **Whisper** | 本地运行，自动丢弃纯音乐视频的无效转录 |
| 说话人分离 | **PyAnnote** (可选) | 区分博主独白 vs 采访对话 |
| 帧提取 | **OpenCV** | 每 2 秒采样 + 场景变化检测 |
| 文字识别 | **PaddleOCR** → **GPT-4o Vision** | 双引擎，自动回退 |
| 文本嵌入 | **BGE-small-zh** (本地) / **OpenAI** | 本地默认，OpenAI key 备选 |
| 向量数据库 | **ChromaDB** | 本地持久化，支持元数据过滤 |
| 关键词检索 | **BM25 (rank-bm25)** | 确保精确查询命中 |
| 混合检索 | **RRF 融合** | 语义 × 关键词 × 推理链 |
| 推理 API | **DeepSeek v4 Flash** (学生) / **DeepSeek v4 Pro** (教师) | 也可替换 OpenAI |
| 交互界面 | **Typer CLI** + **FastAPI** + **Streamlit** | 三种交互方式 |

---

## 许可证

MIT
