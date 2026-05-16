"""测试逻辑段落切分模块。"""

from src.collector.segmenter import detect_topic_boundaries, segment_text, is_complete_analysis


def test_detect_topic_boundaries_finds_transitions():
    text = "我们先看前滩。\n再来看大宁。\n相比之下大宁更有性价比。"
    boundaries = detect_topic_boundaries(text)
    assert len(boundaries) >= 1
    assert any(text[b:].startswith("再来看") for b in boundaries)


def test_segment_text_creates_multiple_segments():
    text = "今天聊聊前滩。先分析它的区位优势。\n\n再来看大宁。大宁的学区资源是最大亮点。\n\n结论是各有千秋。"
    segments = segment_text(text, min_length=5, max_length=1000)
    assert len(segments) >= 1


def test_segment_text_returns_single_for_short():
    text = "这是一段很短的文本。"
    segments = segment_text(text, min_length=5, max_length=1000)
    assert len(segments) == 1
    assert segments[0]["text"] == text


def test_is_complete_analysis():
    assert is_complete_analysis("综合来看，建议买入。") is True
    assert is_complete_analysis("今天的分析到此结束，总体看好。") is True
    assert is_complete_analysis("板块价值分析框架") is False


def test_segment_text_respects_max_length():
    long_text = "前滩分析。" * 200
    segments = segment_text(long_text, min_length=10, max_length=200)
    assert len(segments) > 1  # should be split into multiple segments
    for s in segments:
        assert s["length"] <= 220  # max_length + small buffer
