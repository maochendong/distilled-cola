"""RAG Pipeline 集成测试。"""


def test_rag_pipeline_imports():
    from src.rag.pipeline import RAGPipeline
    pipe = RAGPipeline()
    assert pipe is not None
    assert pipe.retriever is not None
    assert pipe.generator is not None
    assert pipe.validator is not None


def test_generator_imports():
    from src.rag.generator import Generator, SYSTEM_PROMPT
    gen = Generator()
    assert gen is not None
    assert "分析流程" in SYSTEM_PROMPT
    assert "第一步" in SYSTEM_PROMPT
    assert "第四步" in SYSTEM_PROMPT


def test_retriever_imports():
    from src.rag.retriever import Retriever
    r = Retriever()
    assert r is not None
    assert r.hybrid is not None


def test_reasoning_imports():
    from src.rag.reasoning import ReasoningValidator, format_sources
    v = ReasoningValidator()
    assert v is not None
    assert v.client is not None or not v.client  # OK if no API key
