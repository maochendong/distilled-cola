"""测试混合检索器 tokenizer。"""

from src.knowledge_base.hybrid_retriever import tokenize


def test_tokenize_chinese():
    tokens = tokenize("前滩大宁学区房")
    assert len(tokens) > 0
    assert "前" in tokens


def test_tokenize_mixed():
    tokens = tokenize("2024年上海房价走势")
    assert any("2024" == t for t in tokens)


def test_tokenize_non_empty():
    assert len(tokenize("")) == 0
    assert len(tokenize("上海")) > 0
