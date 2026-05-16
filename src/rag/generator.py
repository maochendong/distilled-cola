"""生成器 — 三层提示词架构的专家级回答生成。

系统提示词架构：
  层1 - 身份层：上海房产分析专家的数字分身
  层2 - 分析流程层：四步推演流程
  层3 - 约束层：溯源、置信度、风险声明

用户消息注入：
  - 检索到的知识片段（context）
  - 检索到的推理链（reasoning chains，作为 few-shot 思维范例）
"""

from __future__ import annotations

from src.config import config
from src.llm import chat_client

SYSTEM_PROMPT = """你是一位深耕上海房产市场的资深分析师，拥有以下核心能力：
- 深度理解上海各板块（前滩、大宁、徐汇滨江、北外滩、唐镇等）的价值逻辑
- 熟悉新房认筹、二手房交易、学区政策、限购贷款等全链条规则
- 能结合宏观经济、土地市场、二手房挂牌等数据进行综合判断

## 分析流程（必须严格遵循）

### 第一步：识别关键变量
从用户问题中提取影响房产决策的核心因子：
- 板块定位：内环/中环/外环，规划能级，产业支撑
- 供需关系：新房供应量、二手挂牌量、去化周期
- 政策导向：限购、贷款、学区、房地产税等
- 价格水平：一二手倒挂幅度、单价/总价区间
- 流动性：二手成交活跃度、挂牌到成交周期
- 用户画像：自住/投资、首套/置换、预算上限

### 第二步：调用历史框架
基于检索到的博主分析路径和推理链，提取适用于当前情景的框架。

### 第三步：代入当前数据
结合板块最新成交均价、挂牌量、政策环境，代入框架进行推演。

### 第四步：给出个性化建议
按照用户画像给出操作建议，格式：
```
【短期策略】...
【中期策略】...
【长期逻辑】...
【风险提示】...
```

## 回答准则
1. 每条关键判断必须标注来源依据（引用知识库中的博主观点或数据）
2. 如果检索不到直接相关的框架，明确说明「博主未曾公开讨论该情景，以下为基于其一般原则的模拟推演」
3. 涉及具体价格或数据时，提示可能存在时效性问题
4. 结尾必须包含：「以上分析为基于博主分析框架的模拟推演，不构成投资建议」
5. 回答风格专业而平实，像在微信语音里给朋友分析一样接地气"""


class Generator:
    """基于 RAG 上下文的专家级回答生成器。"""

    def __init__(self) -> None:
        self.client = chat_client()
        self.model = config.student_model

    def generate(
        self,
        query: str,
        context: str = "",
        reasoning_chains: str = "",
        system_prompt: str | None = None,
    ) -> str:
        """根据检索上下文和推理链生成四步结构分析。

        Args:
            query: 用户问题
            context: 检索到的相关知识片段
            reasoning_chains: 检索到的推理链（作为 few-shot 思维范例）
            system_prompt: 可覆盖系统提示词

        Returns:
            结构化分析回答
        """
        if not context and not reasoning_chains:
            return "⚠️ 知识库中暂未找到相关参考信息，请换个角度提问。"

        # 构建用户消息
        user_parts = [f"## 用户问题\n{query}\n"]

        if reasoning_chains:
            user_parts.append(
                f"## 博主历史推理链（供参考其分析框架）\n{reasoning_chains}\n"
            )

        if context:
            user_parts.append(
                f"## 检索到的知识片段\n{context}\n"
            )

        user_parts.append(
            "请严格遵循系统提示中的四步分析流程，给出专业、深入的回答。"
        )

        messages = [
            {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.4,
            max_tokens=2048,
        )
        return resp.choices[0].message.content or ""
