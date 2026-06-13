"""
ProfileRouter — 用户身份路由
根据用户画像决定注入哪种身份提示词
"""

from __future__ import annotations

from typing import Optional

from src.data.user_profiles import get_profile


IDENTITY_MAP = {
    "firsthome": "首套刚需",
    "herhome": "独立女性",
    "family": "家庭改善",
    "golden": "养老置业",
    "general": "通用",
}

IDENTITY_PROMPTS = {
    "firsthome": "注意：用户是首套刚需购房者。回答应侧重通勤便利、总价门槛、首付月供、增值潜力、首套政策优惠。",
    "herhome": "注意：用户是独立女性购房者。回答应侧重安全底线(夜间照明/安保/监控)、产权规划、贷款方案、流动性、独居适配。",
    "family": "注意：用户是有家庭的改善型购房者。回答应侧重学区质量、空间动线、置换链条、社区儿童友好度、双职工通勤平衡。",
    "golden": "注意：用户是养老置业者。回答应侧重医疗保障、无障碍通道、社区支持系统、电梯配置、生活便利度。",
    "general": "",
}


class ProfileRouter:
    """根据用户画像路由到不同的回答策略。"""

    @staticmethod
    def get_identity_context(profile_id: Optional[str]) -> str:
        """获取用户画像对应的身份提示上下文。"""
        if not profile_id:
            return ""
        profile = get_profile(profile_id)
        if not profile:
            return ""
        identity = profile.get("identity_type", "general")
        return IDENTITY_PROMPTS.get(identity, "")

    @staticmethod
    def get_identity_label(profile_id: Optional[str]) -> str:
        """获取身份标签。"""
        if not profile_id:
            return "通用"
        profile = get_profile(profile_id)
        if not profile:
            return "通用"
        identity = profile.get("identity_type", "general")
        return IDENTITY_MAP.get(identity, "通用")

    @staticmethod
    def route_query(query: str, profile_id: Optional[str]) -> str:
        """将身份上下文注入到查询中。"""
        context = ProfileRouter.get_identity_context(profile_id)
        if not context:
            return query
        return f"[用户画像] {context}\n\n问题: {query}"
