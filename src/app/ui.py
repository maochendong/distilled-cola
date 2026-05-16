"""Streamlit 交互界面 — 上海房产分析专家。"""

from __future__ import annotations

import streamlit as st

from src.rag.pipeline import RAGPipeline

st.set_page_config(
    page_title="蒸馏小可乐 — 上海房产分析专家",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 蒸馏小可乐")
st.markdown("**上海房产分析专家** — 基于博主知识蒸馏的四步推演系统")

# 示例问题
example_questions = [
    "800万预算，前滩vs大宁怎么选？",
    "现在是上海买房的好时机吗？",
    "云锦东方真的值得摇吗？",
    "上海认房不认贷后，置换链条怎么走？",
    "徐汇和浦东的学区房，哪个更抗跌？",
]

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("💬 输入你的上海房产问题", placeholder="如：800万预算前滩vs大宁怎么选？")
with col2:
    top_k = st.number_input("检索数量", min_value=1, max_value=10, value=5)

st.markdown("**试试这些问题：**")
cols = st.columns(2)
for i, q in enumerate(example_questions):
    cols[i % 2].button(q, key=f"example_{i}", on_click=lambda q=q: st.session_state.update({"query": q}))

# 自动填充示例
if "query" in st.session_state:
    query = st.session_state.query

if query:
    with st.spinner("🔍 正在检索博主知识库并生成四步分析..."):
        pipe = RAGPipeline()
        result = pipe.ask(query.strip(), top_k=top_k)

    st.markdown("---")
    st.markdown("### 💡 分析")
    st.markdown(result["answer"])

    col1, col2, col3 = st.columns(3)
    col1.metric("置信度", f"{result['confidence']:.0%}")
    col2.metric("推理链引用", result.get("reasoning_chains_used", 0))
    col3.metric("参考来源", len(result["sources"]))

    if result["sources"]:
        with st.expander("📚 参考来源"):
            for s in result["sources"]:
                st.markdown(f"- **{s.get('source', '未知')}** (相关度: {s.get('score', 0):.2f})")
                st.markdown(f"  > {s.get('snippet', '')}")

else:
    st.info("👆 输入你的上海房产相关问题，获取基于博主知识框架的四步分析")

st.markdown("---")
st.caption("⚠️ 以上分析为基于博主分析框架的模拟推演，不构成投资建议")
