# Specification — 蒸馏小可乐

> Product: 上海房产 AI 专家系统 · 四大垂直服务模块
> Last Update: 2026-06-11 (T-001~T-009 已完成)
> Phase: Implement v1.1

---

## 项目当前状态

### 技术架构（基于 README + 源码）
| 层 | 组件 | 状态 |
|----|------|:----:|
| 数据采集 | Whisper 转录 + PaddleOCR + GPT-4o Vision | ✅ |
| 向量存储 | ChromaDB (`sh_knowledge` 单集合) | ✅ |
| 混合检索 | BM25 + 语义 + RRF + Cross-encoder 精排 | ✅ |
| 实时搜索 | AnySearch API (domain=home, zone=cn, 上海限定) | ✅ |
| 问答引擎 | DeepSeek Flash 生成 + Pro 教师验证 | ✅ |
| CLI | `cola` (Typer, 9 子命令) | ✅ |
| API | FastAPI (`/ask`, `/stats`, `/health`) | ✅ |
| UI | Streamlit (蓝白主题) | ✅ |
| 部署 | Docker (Nginx + Streamlit) | ✅ |
| **差异化** | 多轮对话 + 流式 + 反馈 + 板块对比 | ✅ T-001~009 |
| 知识库 | 1 位博主, 230 视频, OCR + 转录双路 | ⏳ 待扩展 |

### 已完成 (T-001~T-009)
| # | 任务 | 文件 |
|:-:|------|------|
| T-001 | 多轮对话支持 | `src/rag/conversation.py` |
| T-002 | 流式输出 | `src/rag/generator.py` (+`generate_stream`) |
| T-003 | 用户反馈闭环 | `src/app/feedback.py` |
| T-004 | 自检循环优化 | `src/rag/validator.py` |
| T-005 | 搜索失败告警 | `src/rag/retriever.py` (+`_last_search_status`) |
| T-006 | Cross-encoder 缓存日志 | `src/rag/reranker.py` |
| T-007 | Web UI 重构 | `src/app/ui.py` (蓝白主题 + 对话式) |
| T-008 | 移动端适配 | `src/app/ui.py` (响应式 CSS) |
| T-009 | 板块对比模块 | `src/app/comparison.py` |

---

## Overview

71 项开发任务，按 6 个 Phase 组织。P0 优先启动（5 项），P1 核心构建（39 项），P2 进阶优化（16 项），P3 商业化（11 项）。**已完成 9 项。**

---

## Phase 1: MVP 优化 (1-2 weeks) ✅ 完成

### P0 — 已交付
- **T-001** ✅ 多轮对话支持 — `conversation_id` + 上下文管理
- **T-002** ✅ 流式输出 — SSE streaming，首 token 延迟优化

### P1 — 高优
- **T-003** 用户反馈闭环 — 👍/👎/纠错按钮，数据飞轮
- **T-004** 自检循环优化 — 最大 3 次指数退避，阈值可配置
- **T-005** 实时搜索失败告警 — 非静默降级，用户有感知

### P2 — 中优
- **T-006** Cross-encoder 缓存日志 — fallback 状态日志

---

## Phase 2: 核心体验提升 (1 month)

### P0 — 立即启动
- **T-007** Web UI 前端重构 — Vue/React SPA，流式渲染，响应式
- **T-008** 移动端适配 / PWA — 自适应布局 + 离线缓存
- **T-009** 板块对比模块 — 2-3 板块并排对比（均价/学区/通勤/增值）

### P1 — 高优
- **T-010** 数据可视化 — ECharts 价格走势/雷达图/热力图
- **T-011** 多模式回答 — 一句话结论/对比表格/投资摘要/详细分析
- **T-012** 房贷计算器 — 月供/利息/税费实时计算
- **T-013** 板块评分系统 — 多因子加权评分
- **T-014** 问答缓存优化 — 语义相似缓存，QPS 5x
- **T-015** **[通用] 用户画像采集 + 身份路由** — 服务差异化第一步
- **T-016** **[通用] 需求深度问卷** — 按身份动态生成追问
- **T-017** **[FirstHome] 首套刚需专题知识注入** — 通勤/总价/上车逻辑
- **T-018** **[FamilyUpgrade] 家庭改善专题知识注入** — 学区/户型/置换

### P2 — 中优
- **T-019** **[HerHome] 安全维度知识注入** — 夜间照明/安保/监控知识
- **T-020** 高级搜索筛选器 — 价格/板块/房型/标签过滤
- **T-021** 智能搜索联想 — 板块名自动推荐相关问题

---

## Phase 3: 数据基础建设 (2-3 months)

### P1 — 高优
- **T-022** 楼盘级数据库 — 链家/贝壳 API，户型/单价/历史成交
- **T-023** 土地市场数据 — 土拍结果/楼板价/新房指导价
- **T-024** 学区数据库 — 划片/学校梯队/对口变化历史
- **T-025** 地铁/规划数据 — 在建地铁/TOD/商业配套
- **T-026** 多博主蒸馏 — 从 1 位扩展到 3-5 位
- **T-027** **[FirstHome] 首套刚需专项数据** — 通勤成本/总价门槛/首付方案
- **T-028** **[FamilyUpgrade] 家庭改善专项数据** — 学校梯队/儿童配套/置换供需
- **T-029** **[HerHome] 女性专属数据收集** — 安全评分/贷款/产权法规
- **T-030** **[GoldenYears] 养老生活圈数据** — 医院/药店/菜场/社区食堂 POI
- **T-031** **[GoldenYears] 物业适老能力评级** — 独居寻访/无障碍/防滑

### P2 — 中优
- **T-032** 宏观经济指标 — GDP/人口/M2/LPR 加入推理上下文
- **T-033** **[GoldenYears] 上海养老设施数据库** — 老年活动中心/大学/诊所
- **T-034** 知识盲区检测 — LLM 定期扫描，生成完整度报告

---

## Phase 4: 高级功能 (3-4 months)

### P1 — 高优
- **T-035** 价格异动提醒与订阅报告 — 涨跌>5% 推送 + 楼市简报
- **T-036** 关注/收藏功能 — 关注板块/楼盘，动态主动推送
- **T-037** 多模态搜索 — 户型图/照片 → OCR+Vision 分析
- **T-038** **[FirstHome] 通勤与增值评分** — 「上车友好度」综合指数
- **T-039** **[FirstHome] 上车决策支持** — 总价门槛/首付方案/时机判断
- **T-040** **[HerHome] 安全维度评分卡** — 夜间照明/安保/监控/底商 ⭐
- **T-041** **[HerHome] 女性贷款/产权规划问答** — 婚前财产/继承/出资方式
- **T-070** **[HerHome] 独居适配度报告** — 户型隐私性/入户动线/快递收纳/紧急呼叫路径
- **T-042** **[FamilyUpgrade] 学区与社区评分** — 「家庭宜居指数」
- **T-043** **[FamilyUpgrade] 改善置换链条规划** — 估值→预算→税费→过渡
- **T-044** **[GoldenYears] 养老适配度评估** — 「养老适配度」综合评分
- **T-045** **[GoldenYears] 子女协作看房模式** — 远程评估报告+三方共享

### P2 — 中优
- **T-046** **[HerHome] 流动性评估模型** — 5-10 年二手流通性预测
- **T-047** **[GoldenYears] 置换链条规划** — 资金衔接/税费优惠/过渡方案
- **T-048** 专家咨询入口 — AI 分析后一键预约真人咨询
- **T-049** 趋势预测 — 历史成交+土地+宏观 → 涨跌分析
- **T-050** 语音输入 — Whisper + 前端录音

---

## Phase 5: 技术架构完善 (ongoing)

### P1 — 高优
- **T-051** A/B 测试框架 — top_k/reranker/温度 参数优化
- **T-052** 监控与日志体系 — Grafana/Prometheus 核心指标
- **T-053** **[通用] 多身份 Prompt 模板系统** — 5 套独立 prompt，路由选择
- **T-054** **[通用] 用户身份 Agent 路由模块** — `ProfileRouter`

### P2 — 中优
- **T-055** 知识库增量更新自动触发 — 文件监听 + Webhook
- **T-056** 知识库版本管理与回滚 — 快照 + 一键回滚
- **T-057** **[通用] 评分器框架抽象** — 统一 `Scorer` 接口
- **T-058** **[通用] 多维度综合评分引擎** — 可调权重「置业匹配度」
- **T-059** 国际化支持 — 英文版界面 + English response

---

## Phase 6: 商业化探索 (on-demand)

### P3 — 远期
- **T-060** 付费报告生成 — PDF 「购房指南」「板块深度白皮书」
- **T-061** 新盘导流 — 展示新盘 + 预约看房分润
- **T-062** 会员体系 — 免费 3 次/天 vs Pro 无限
- **T-063** B2B API 服务 — 话术查询/竞品分析/价值白皮书
- **T-064** **[B2B] FirstHome 刚需赋能工具** — 评分卡/通勤报告/首付 SOP
- **T-065** **[B2B] HerHome 经纪人赋能工具** — 安全评分卡/产权话术/贷款 SOP
- **T-066** **[B2B] FamilyUpgrade 改善赋能工具** — 学区评分卡/置换工具/通勤 SOP
- **T-067** **[B2B] GoldenYears 适老置业工具** — 适老评分/生活圈报告/协作 SOP
- **T-068** 内容矩阵 — 小红书/公众号自动生成
- **T-071** 垂直人群社区 — HerHome/FirstHome/FamilyUpgrade/GoldenYears 专属社群
- **T-072** 服务商生态 — 按身份推荐律师/装修/搬家/适老化改造

---

## Milestones

| Milestone | Tasks | Target |
|-----------|-------|--------|
| M1: MVP 基础 | T-001 ~ T-006 | Week 2 |
| M2: 服务差异化就绪 | T-007 ~ T-021（含画像+问卷） | Month 1 |
| M3: 数据底座 | T-022 ~ T-034（四大模块数据全部入库） | Month 2-3 |
| M4: 差异化能力上线 | T-035 ~ T-050（评分器+决策支持） | Month 3-4 |
| M5: 架构成熟 | T-051 ~ T-059（路由/模板/框架） | Ongoing |
| M6: 商业闭环 | T-060 ~ T-068（B2B + 会员） | On-demand |

---

## 附录 A: 源码文件清单 (2026-06-11)

> 后续迭代修改文件前必须参考此清单，避免重复造轮子或破坏现有功能。

### 数据采集层 (`src/collector/`)
| 文件 | 功能 | 依赖 |
|------|------|------|
| `pipeline.py` | 视频采集管道入口 | transcriber, frame_extractor, ocr, segmenter, annotator |
| `transcriber.py` | Whisper 转录 + 字幕流提取 | ffmpeg, whisper |
| `frame_extractor.py` | OpenCV 帧提取 (2s采样) | opencv |
| `ocr.py` | 双引擎 OCR (PaddleOCR + GPT-4o Vision) | paddleocr, openai |
| `segmenter.py` | 逻辑段落切分 (话题转移检测) | - |
| `annotator.py` | 房产领域双维度标注 (实体/逻辑/建议) | DeepSeek Pro |

### 知识库层 (`src/knowledge_base/`)
| 文件 | 功能 |
|------|------|
| `vector_store.py` | ChromaDB wrapper (collection管理, 检索) |
| `embedder.py` | bge-m3 / OpenAI embedding |
| `hybrid_retriever.py` | BM25 + 语义 + RRF 混合检索 |
| `reranker.py` | Cross-encoder 精排 (BGE Reranker v2) |
| `reasoning_index.py` | 推理链索引 (板块+逻辑标签匹配加分) |

### RAG 管线 (`src/rag/`)
| 文件 | 功能 | 状态 |
|------|------|:----:|
| `pipeline.py` | ask() 编排 (检索→精排→生成→验证) | v2.0 |
| `retriever.py` | 检索器封装 (搜索状态追踪) | v2.0 |
| `generator.py` | DeepSeek 生成 (三层提示词+流式) | v2.0 |
| `reasoning.py` | 教师验证 + 溯源格式化 | 生产 |
| `web_search.py` | AnySearch 实时搜索客户端 | 生产 |
| `validator.py` | 指数退避自检验证 | [NEW] |
| `conversation.py` | 多轮对话管理 | [NEW] |
| `reranker.py` | 非阻塞精排 (fallback日志) | [NEW] |
| `config.py` | RAG 配置 (可能重复) | ⚠️ 注意与 `src/config.py` 区分 |

### 应用层 (`src/app/`)
| 文件 | 功能 | 状态 |
|------|------|:----:|
| `cli.py` | Typer CLI (`cola` 9个子命令) | 生产 |
| `api.py` | FastAPI (`/ask`, `/stats`, `/health`) | 生产 |
| `ui.py` | Streamlit UI (蓝白主题+对话+对比) | v3.0 |
| `feedback.py` | 用户反馈 JSONL 存储 | [NEW] |
| `comparison.py` | 板块对比引擎+ECharts 雷达图 | [NEW] |

### 根级文件
| 文件 | 功能 |
|------|------|
| `src/config.py` | 主配置 (DeepSeek/ChromaDB/路径/env) |
| `src/llm.py` | LLM 客户端封装 |
| `pyproject.toml` | 依赖管理 + CLI入口 (`cola`) |
| `Dockerfile` | 多阶段构建 (BuildKit cache, CPU-only torch) |
| `docker-compose.yml` | Nginx:80 + Streamlit:8501, HF 模型缓存卷 |

### 脚本 (`scripts/`)
| 文件 | 功能 |
|------|------|
| `ingest_all.py` | OCR 数据全量蒸馏入库 |
| `ingest_transcriptions.py` | 转录数据增量追加入库 (带检查点) |
| `batch_ocr_video.py` | 批量帧 OCR (macOS Vision) |
| `batch_transcribe_mlx.py` | 批量 mlx-whisper 转录
