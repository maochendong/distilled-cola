"""Streamlit 交互界面 — 上海房产分析专家"""

from __future__ import annotations

import os
import streamlit as st

from src.rag.pipeline import RAGPipeline
from src.rag.conversation import conversation_manager
from src.app.feedback import feedback_handler, Feedback
from src.app.comparison import ComparisonEngine


st.set_page_config(
    page_title="蒸馏小可乐 — 上海房产分析专家",
    page_icon="🏠",
    layout="centered",
)

# ── 蓝白主题 CSS ──
st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #F0F4FF 0%, #FFFFFF 100%); }
    .main .block-container { max-width: 860px; padding-top: 2.5rem; }
    .stChatMessage { background: white; border-radius: 16px; padding: 1rem 1.2rem; border: 1px solid #E2E8F0; margin-bottom: 0.8rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
    div[data-testid="stChatMessage"] { background: white; }
    .stButton > button { border-radius: 10px; font-weight: 500; }
    .stButton > button[kind="primary"] { background: #2563EB; }
    section[data-testid="stSidebar"] { background: white; border-right: 1px solid #E2E8F0; }
    @media (max-width: 768px) {
        .main .block-container { padding: 1rem !important; max-width: 100%; }
        .stButton button { width: 100%; }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


def init_session():
    if "conv_id" not in st.session_state:
        st.session_state.conv_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []


init_session()


# ════════════ 侧边栏 ════════════
with st.sidebar:
    st.markdown("## 👤 身份选择")
    st.radio(
        "选择你的购房身份",
        ["❓ 通用", "🏠 首套刚需", "🏡 独立女性", "👨‍👩‍👧‍👦 家庭改善", "🌅 养老置业"],
        key="identity_radio",
    )
    st.markdown("---")
    st.markdown("## 📊 板块对比")
    districts = st.multiselect(
        "选择 2-3 个板块",
        ["前滩", "大宁", "徐汇滨江", "北外滩", "唐镇", "古北", "联洋", "碧云", "森兰", "新江湾", "华漕", "南翔", "泗泾", "九亭", "江桥"],
        max_selections=3, placeholder="点击选择...",
    )
    if len(districts) >= 2 and st.button("开始对比", type="primary", use_container_width=True):
        with st.spinner("生成对比分析..."):
            engine = ComparisonEngine(pipeline=get_pipeline())
            result = engine.compare(districts)
            st.markdown("### 📊 对比结果")
            st.markdown(engine.render_comparison_table(result))
            chart_data = engine.render_radar_chart_data(result)
            if chart_data and chart_data != "{}":
                st.components.v1.html(
                    f'<div id="r" style="height:380px;"></div>'
                    f'<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>'
                    f'<script>var c=echarts.init(document.getElementById("r"));c.setOption({chart_data});window.addEventListener("resize",()=>c.resize());</script>',
                    height=400,
                )
    st.markdown("---")
    st.markdown("### ℹ️ 关于")
    st.caption("蒸馏小可乐 将房产博主专业知识「蒸馏」为 AI 分析系统。基于四步推演回答你的上海房产问题。")
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conv_id = None
        st.rerun()


# ════════════ 主界面 ════════════

# 欢迎状态：搜索引擎式简约首页
if not st.session_state.messages:
    st.markdown("<div style='height:8rem'></div>", unsafe_allow_html=True)

    logo_path = "logo.jpeg"
    if os.path.exists(logo_path):
        _, c, _ = st.columns([2, 1, 2])
        with c:
            st.image(logo_path, width=120)

    st.markdown(
        "<h1 style='text-align:center; font-size:2.2rem; margin-bottom:0.3rem;'>蒸馏小可乐</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:#64748B; font-size:1.05rem; margin-bottom:2rem;'>助你决策人生最大一笔投资</p>",
        unsafe_allow_html=True,
    )

    col_q, col_btn = st.columns([5, 1])
    with col_q:
        query = st.text_input(
            "💬",
            placeholder="输入你的上海房产问题，如：800万预算前滩vs大宁怎么选？",
            label_visibility="collapsed",
            key="welcome_query",
        )
    with col_btn:
        submitted = st.button("🔍 分析", type="primary", use_container_width=True)

    if submitted and query.strip():
        st.session_state.messages.append({"role": "user", "content": query.strip()})
        st.rerun()

# 对话状态：聊天界面
if st.session_state.messages:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "answer_id" in msg:
                c1, _ = st.columns([1, 18])
                aid = msg["answer_id"]
                q = msg.get("query", "")
                if c1.button("👍", key=f"up_{aid}"):
                    feedback_handler.save_feedback(Feedback(answer_id=aid, rating="up", query=q))
                    st.toast("感谢反馈！", icon="✅")
                if c1.button("👎", key=f"dn_{aid}"):
                    feedback_handler.save_feedback(Feedback(answer_id=aid, rating="down", query=q))
                    st.toast("已记录", icon="📝")

    prompt = st.chat_input("继续提问...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

    # 生成新回答
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        q = st.session_state.messages[-1]["content"]
        if not st.session_state.conv_id:
            conv = conversation_manager.create()
            st.session_state.conv_id = conv.id

        with st.chat_message("assistant"):
            placeholder = st.empty()
            pipeline = get_pipeline()
            full, answer_id, confidence = "", None, 1.0
            for chunk in pipeline.ask_stream(query=q, top_k=5, conv_id=st.session_state.conv_id):
                if chunk["type"] == "token":
                    full += chunk["content"]
                    placeholder.markdown(full + "▌")
                elif chunk["type"] in ("warning", "error"):
                    full += chunk["content"]
                    placeholder.markdown(full + "▌")
                elif chunk["type"] == "done":
                    answer_id = chunk.get("answer_id")
                    confidence = chunk.get("confidence", 1.0)
                    if isinstance(confidence, float):
                        full += f"\n\n> 🎯 置信度: {confidence:.0%}"
                    placeholder.markdown(full)

            c1, c2, c3 = st.columns(3)
            c1.metric("置信度", f"{confidence:.0%}")
            c2.metric("对话轮次", f"{len(st.session_state.messages)//2 + 1}")
            c3.metric("检索量", "5 条")

        st.session_state.messages.append({
            "role": "assistant", "content": full,
            "answer_id": answer_id, "query": q,
        })
        st.rerun()

st.markdown("---")
st.caption("⚠️ 以上分析为基于博主分析框架的模拟推演，不构成投资建议")
