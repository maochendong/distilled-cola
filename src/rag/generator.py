"""
Generator — DeepSeek 答案生成器
支持流式输出 (T-002) 和对话上下文注入 (T-001)
"""
import logging
from typing import Optional, Generator

logger = logging.getLogger(__name__)

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
1. 每条关键判断必须标注来源依据，格式为 **句末标注来源编号** 如 `[1]`、`[2]`，编号对应上下文片段序号
2. 区分两类来源：`[x]` 来自博主知识库（历史分析框架），`[实时]` 来自网络搜索（当前行情数据）
3. 如果检索不到直接相关的框架，明确说明「博主未曾公开讨论该情景，以下为基于其一般原则的模拟推演」
4. **实时数据可能有延迟**，涉及具体价格或数据时提示可能存在时效性差异
5. 结尾必须包含：「以上分析为基于博主分析框架的模拟推演，不构成投资建议」
6. 回答风格专业而平实，像在微信语音里给朋友分析一样接地气"""


class Generator:
    """答案生成器 — 支持流式和阻塞两种模式"""

    def __init__(self, client=None, model: str = None,
                 system_prompt: str = None,
                 temperature: float = 0.4,
                 max_tokens: int = 8192):
        from src.llm import chat_client
        from src.config import config
        self.client = client or chat_client()
        self.model = model or config.student_model
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, query: str, context: str = "",
                 reasoning_chains: str = "",
                 system_prompt: Optional[str] = None,
                 conversation_context: str = "",
                 web_context: str = "") -> str:
        """阻塞式生成"""
        if not context and not reasoning_chains and not web_context:
            return "⚠️ 知识库中暂未找到相关参考信息，请换个角度提问。"
        messages = self._build_messages(
            query, context, reasoning_chains,
            system_prompt, conversation_context, web_context
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.error("生成失败: %s", e)
            raise

    def generate_stream(self, query: str, context: str = "",
                        reasoning_chains: str = "",
                        system_prompt: Optional[str] = None,
                        conversation_context: str = "",
                        web_context: str = "") -> Generator[str, None, None]:
        """流式生成 — 逐 token 产出 (T-002)"""
        if not context and not reasoning_chains and not web_context:
            yield "⚠️ 知识库中暂未找到相关参考信息，请换个角度提问。"
            return
        messages = self._build_messages(
            query, context, reasoning_chains,
            system_prompt, conversation_context, web_context
        )
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content
        except Exception as e:
            logger.error("流式生成失败: %s", e)
            yield f"⚠️ 生成错误: {e}"

    def _build_messages(self, query: str, context: str,
                        reasoning_chains: str,
                        system_prompt: Optional[str],
                        conversation_context: str,
                        web_context: str = "") -> list:
        """组装消息列表"""
        prompt = system_prompt or self.system_prompt

        user_parts = [f"## 用户问题\n{query}\n"]
        if conversation_context:
            user_parts.append(f"{conversation_context}\n")
        if web_context:
            user_parts.append(f"{web_context}\n")
        if reasoning_chains:
            user_parts.append(
                f"## 博主历史推理链（供参考）\n{reasoning_chains}\n"
            )
        if context:
            user_parts.append(f"## 检索到的知识片段\n{context}\n")
        user_parts.append(
            "请严格遵循系统提示中的四步分析流程，给出专业、深入的回答。"
        )

        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]
