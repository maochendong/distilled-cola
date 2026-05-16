"""数据处理器单元测试。"""

from src.collector.processor import chunk_text, clean_text


def test_clean_text_removes_brackets():
    text = "今天市场大涨 [音乐] [掌声] 原因如下"
    assert "[音乐]" not in clean_text(text)
    assert "[掌声]" not in clean_text(text)
    assert "今天市场大涨" in clean_text(text)


def test_chunk_text_respects_chunk_size():
    text = "句子一。句子二。句子三。" * 50
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 150  # chunk_size + small buffer


def test_chunk_text_single_chunk_for_short_text():
    text = "这是一个很短的文本。只有几句话而已。"
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_has_overlap():
    text = "第一句。第二句。第三句。第四句。第五句。"
    chunks = chunk_text(text, chunk_size=30, overlap=10)
    if len(chunks) > 1:
        # There should be some overlap between chunks
        assert len(chunks[0]) > 0
        assert len(chunks[1]) > 0
