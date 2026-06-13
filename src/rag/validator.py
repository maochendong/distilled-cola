"""
ReasoningValidator — 教师-学生验证体系
支持可配置阈值、指数退避重试、JSON 结构化输出
"""
import json
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

SELF_CHECK_PROMPT = """你是上海房产分析回答的质量检查员。

请按以下 6 条标准逐一核验回答质量，以 JSON 格式输出结果：

1. 是否识别了关键变量（板块/预算/政策/供需等）？
2. 是否有推理链条（分析从因到果，而非直接下结论）？
3. 是否有风险提示（潜在的局限性和风险）？
4. 是否有具体建议（根据用户情况给出可操作指引）？
5. 是否有来源标注（关键判断标注了 [1]、[2] 等来源编号）？
6. 来源引用是否真实（检查引用内容是否能在来源中找到依据）？

用户问题: {query}
回答: {answer}

输出格式:
{{
    "passed_checks": ["变量识别", "推理链条", "风险提示"],
    "failed_checks": ["来源标注"],
    "confidence": 0.72,
    "reason": "缺少来源编号，但分析框架完整"
}}
"""


class ValidationResult:
    """验证结果"""
    def __init__(self, passed_checks: list = None, failed_checks: list = None,
                 confidence: float = 1.0, reason: str = ""):
        self.passed_checks = passed_checks or []
        self.failed_checks = failed_checks or []
        self.confidence = confidence
        self.reason = reason

    def needs_refinement(self, threshold: float = 0.6, max_failures: int = 2) -> bool:
        """判断是否需要优化回答"""
        return self.confidence < threshold or len(self.failed_checks) >= max_failures

    @classmethod
    def from_dict(cls, data: dict) -> "ValidationResult":
        return cls(
            passed_checks=data.get("passed_checks", []),
            failed_checks=data.get("failed_checks", []),
            confidence=data.get("confidence", 1.0),
            reason=data.get("reason", ""),
        )


class ReasoningValidator:
    """教师模型验证器 — 可配置阈值和重试策略"""

    def __init__(self, client, teacher_model: str,
                 confidence_threshold: float = 0.6,
                 max_failures: int = 2,
                 max_retries: int = 3,
                 base_delay: float = 1.0):
        self.client = client
        self.teacher_model = teacher_model
        self.confidence_threshold = confidence_threshold
        self.max_failures = max_failures
        self.max_retries = max_retries
        self.base_delay = base_delay

    def validate(self, query: str, answer: str,
                 sources: Optional[list] = None) -> ValidationResult:
        """对回答进行质量验证，支持指数退避重试"""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.teacher_model,
                    messages=[
                        {"role": "system",
                         "content": "你是上海房产分析回答的质量检查员。"},
                        {"role": "user",
                         "content": SELF_CHECK_PROMPT.format(
                             query=query, answer=answer)},
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                data = json.loads(resp.choices[0].message.content)
                return ValidationResult.from_dict(data)

            except Exception as e:
                last_error = e
                logger.warning(
                    "验证尝试 %d/%d 失败: %s",
                    attempt + 1, self.max_retries, e
                )
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    logger.info("等待 %.1fs 后重试...", delay)
                    time.sleep(delay)

        logger.error("验证在 %d 次尝试后全部失败: %s",
                     self.max_retries, last_error)
        return ValidationResult(
            confidence=0.5,
            reason=f"验证服务暂时不可用（{last_error}）",
        )

    def needs_refinement(self, result: ValidationResult) -> bool:
        """判断是否需要优化回答"""
        return result.needs_refinement(
            threshold=self.confidence_threshold,
            max_failures=self.max_failures,
        )
