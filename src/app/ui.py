"""Streamlit 交互界面 — 上海房产分析专家。"""

from __future__ import annotations

import streamlit as st

from src.rag.pipeline import RAGPipeline


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()

st.set_page_config(
    page_title="蒸馏小可乐 — 上海房产分析专家",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 蒸馏小可乐")
st.markdown("**上海房产分析专家** — 基于博主知识蒸馏的四步推演系统")

# 真实感较高的示例问题
example_questions = [
    "总价600万，杨浦vs普陀vs闵行，哪里的老公房更保值？",
    "浦东唐镇和金桥，自住+保值，哪里更值得上车？",
    "外环外放开限购后，青浦新城和嘉定新城怎么选？",
    "2000万预算，黄浦老西门vs徐汇滨江，哪个长期潜力更大？",
    "大虹桥辐射区中，华漕vs徐泾vs江桥，自住改善选哪里？",
]

col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    query = st.text_input(
        "💬 输入你的上海房产问题",
        placeholder="如：800万预算前滩vs大宁怎么选？",
    )
with col2:
    top_k = st.number_input("检索数量", min_value=1, max_value=10, value=5)
with col3:
    submitted = st.button("🔍 分析", type="primary", use_container_width=True)

st.markdown("**试试这些问题：**")
cols = st.columns(2)
for i, q in enumerate(example_questions):
    if cols[i % 2].button(q, key=f"example_{i}", use_container_width=True):
        submitted = True
        query = q


@st.cache_data(ttl=3600, show_spinner=False)
def ask_question(q: str, k: int) -> dict:
    return get_pipeline().ask(q.strip(), top_k=k)


if submitted and query:
    with st.spinner("🔍 正在检索博主知识库并生成四步分析..."):
        result = ask_question(query.strip(), top_k)

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

elif "answer" not in st.session_state:
    st.info("👆 输入你的上海房产相关问题，获取基于博主知识框架的四步分析")

st.markdown("---")
st.caption("⚠️ 以上分析为基于博主分析框架的模拟推演，不构成投资建议")
