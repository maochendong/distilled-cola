# Project Constitution — 蒸馏小可乐

## Vision
上海房产 AI 专家系统，将房产博主专业知识「蒸馏」为可交互的决策支持系统——从「以房源为中心」转向「以人的生命周期为中心」，按购房者身份和人生阶段提供差异化服务。

## Core Values
1. **以人为中心**：不按房源品类分，按人的身份和人生阶段分
2. **数据驱动**：每条回答有据可查，溯源可验证
3. **差异优先**：不做好大一统的通用答案，做好垂直人群的精准服务
4. **渐进交付**：按 Phase 分步落地，每个 Phase 可独立验收
5. **开箱即用**：纯 Python 3 实现，零外部依赖

## Technology Stack (from README + pyproject.toml)
- **Language**: Python 3.9+
- **Embedding**: BGE-M3 (本地 1024D) / OpenAI (备选)
- **Vector DB**: ChromaDB (本地持久化, HNSW, cosine)
- **Hybrid Retrieval**: BM25 (rank-bm25 + jieba) + 语义 (bge-m3) + RRF 融合
- **Reranker**: BGE Reranker v2 (Cross-encoder, 未缓存时跳过不阻塞)
- **LLM**: DeepSeek V4 Flash (学生, generation) + DeepSeek V4 Pro (教师, validation)
- **Realtime Search**: AnySearch API (domain=home, zone=cn, 上海限定)
- **Data Pipeline**: Whisper/mxl-whisper 转录 + PaddleOCR/GPT-4o Vision OCR
- **CLI**: Typer (`cola` command, 9 subcommands)
- **API**: FastAPI (`/ask`, `/stats`, `/health` + StaticFiles)
- **UI**: Streamlit (蓝白主题) + 自建 Web UI
- **Deployment**: Docker (Nginx:80 → Streamlit:8501, HF 模型缓存持久化)

## Quality Standards
- **Test Coverage**: 核心 RAG 管线 ≥ 80%
- **Code Review**: 所有 PR 必须 review
- **Documentation**: 每个模块有 README，API 有 OpenAPI spec
- **验证**: 教师-学生验证体系确保回答质量

## Development Principles
1. **服务差异化优先**：用户画像 → 身份路由 → 评分器，先于通用功能
2. **复用现有管线**：新知识线复用已有视频蒸馏管道，不重复造轮子
3. **渐进增强**：每个 Phase 产出可独立演示的增量
4. **数据先行**：评分器的精度取决于底层数据质量，数据采集优先于评分 UI
5. **B2B 最后做**：先跑通 C 端价值，再考虑商业化输出

## Constraints
- **Must have**: 视频知识蒸馏管道、RAG 问答、用户画像系统、四大垂直模块
- **Must not have**: 不依赖任何付费 API（除 LLM 调用外）
- **Performance**: 单次问答 < 10s，首 token < 3s
- **Security**: API Key 不提交到 Git，环境变量注入
