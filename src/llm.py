"""统一 LLM 客户端工厂 — 管理 DeepSeek / OpenAI 等多后端。

使用策略：
  - Chat（生成/标注/质检）：DeepSeek（deepseek-v4-flash / deepseek-v4-pro）
  - Embedding：本地 sentence-transformers（默认），OpenAI 备选
  - Vision（图片理解）：OpenAI（DeepSeek 不支持多模态）
"""

from __future__ import annotations

from openai import OpenAI

from src.config import config


def chat_client() -> OpenAI | None:
    """获取 Chat 客户端（DeepSeek，用于问答生成/标注/质检）。

    使用 OpenAI SDK + DeepSeek base_url，因为 DeepSeek 兼容 OpenAI API。
    """
    api_key = config.deepseek_api_key or config.openai_api_key
    if not api_key:
        return None

    if config.deepseek_base_url:
        return OpenAI(api_key=api_key, base_url=config.deepseek_base_url)

    kwargs: dict = {"api_key": api_key}
    # 如果用的是 OpenAI key，尝试识别模型名是否为 deepseek 模型
    if "sk-" + "" == api_key[:3] if len(api_key) > 3 else False:
        pass  # OpenAI key, 用默认 endpoint
    return OpenAI(**kwargs)


def vision_client() -> OpenAI | None:
    """获取 Vision 客户端（OpenAI，用于图片文字提取）。"""
    if config.openai_api_key:
        return OpenAI(api_key=config.openai_api_key)
    return None


def embedding_client() -> OpenAI | None:
    """获取 Embedding 客户端（OpenAI，用于向量嵌入）。"""
    if config.openai_api_key:
        return OpenAI(api_key=config.openai_api_key)
    return None


def chat(model: str | None = None, **kwargs) -> dict | None:
    """快捷调用 Chat 补全，返回响应字典。"""
    client = chat_client()
    if not client:
        return None
    model = model or config.student_model
    try:
        resp = client.chat.completions.create(model=model, **kwargs)
        return {
            "content": resp.choices[0].message.content or "",
            "model": resp.model,
            "usage": {
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            } if resp.usage else {},
        }
    except Exception as e:
        raise RuntimeError(f"LLM 调用失败 (model={model}): {e}") from e
