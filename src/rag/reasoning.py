"""推理自检与溯源模块 — 确保回答质量和可靠性。"""

from __future__ import annotations

import json
from typing import Any

from src.config import config
from src.llm import chat_client

SELF_CHECK_PROMPT = """请对以下关于上海房产的分析回答进行质量检查。

## 回答
{answer}

## 用户问题
{query}

## 检查清单
1. **是否识别了关键变量**？板块/预算/政策/供需等核心因子都覆盖了吗？
2. **是否有推理链条**？分析是从因到果，还是直接下结论？
3. **是否有风险提示**？是否提到了潜在的局限性和风险？
4. **是否有具体建议**？是否根据用户情况给出了可操作的指引？
5. **是否有来源标注**？关键判断是否标注了 [1]、[2] 等来源编号，且与实际上下文匹配？
6. **来源引用是否真实**？检查回答引用的内容是否能在给定来源中找到依据；如果声称某某博主说过某观点，但可用来源中查无此据，记为失败

## 输出格式
```json
{{
  "passed_checks": ["...", "..."],
  "failed_checks": ["...", "..."],
  "confidence": 0.0-1.0,
  "missing_info": "如果置信度低于 0.7，说明缺少什么信息",
  "suggestions": "如何改进回答"
}}
```"""


class ReasoningValidator:
    """回答质量验证器。"""

    def __init__(self) -> None:
        self.client = chat_client()

    def validate(self, query: str, answer: str, sources: list[dict] | None = None) -> dict:
        """对回答进行质量检查。"""
        if not self.client or not answer:
            return {"confidence": 0.0, "passed_checks": [], "failed_checks": ["无回答可供检查"]}

        # 附加来源信息供溯源校验
        source_context = ""
        if sources:
            parts = []
            for i, s in enumerate(sources, 1):
                snippet = (s.get("snippet", "") or "")[:150]
                source = s.get("source", "未知")
                parts.append(f"[{i}] {source}: {snippet}")
            source_context = "\n".join(parts)

        user_msg = SELF_CHECK_PROMPT.format(query=query, answer=answer)
        if source_context:
            user_msg += f"\n\n## 可用参考来源（供核对引用）\n{source_context}"

        try:
            resp = self.client.chat.completions.create(
                model=config.teacher_model,
                messages=[
                    {"role": "system", "content": "你是上海房产分析回答的质量检查员。只输出 JSON。"},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception as e:
            return {"confidence": 0.0, "passed_checks": [], "failed_checks": [str(e)]}

    def needs_refinement(self, result: dict) -> bool:
        """判断是否需要优化回答。"""
        return (
            result.get("confidence", 1.0) < 0.6
            or len(result.get("failed_checks", [])) >= 2
        )


def format_sources(hits: list[dict]) -> str:
    """将检索结果格式化为可读的来源列表。"""
    if not hits:
        return "无检索来源"

    lines = ["📚 参考来源："]
    for h in hits:
        source = ""
        if "metadata" in h and h["metadata"]:
            source = h["metadata"].get("source", h["metadata"].get("title", ""))
        elif "trigger" in h:
            source = f"推理链: {h['trigger'][:60]}"

        score = h.get("score", 0)
        lines.append(f"  • [{source}] (相关度: {score:.2f})")

        # snippet 预览
        text = h.get("text", "")
        if text:
            lines.append(f"    {text[:100]}...")

    return "\n".join(lines)
