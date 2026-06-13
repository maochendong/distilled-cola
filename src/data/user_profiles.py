"""
User Profile — 用户画像 CRUD
"""

from __future__ import annotations

import json
import uuid
from typing import Optional

from src.data.db import get_db, query, query_one, execute


def create_profile(identity_type: str = "general",
                   identity_label: str = "",
                   questionnaire: dict = None) -> dict:
    """创建用户画像。"""
    profile_id = str(uuid.uuid4())[:12]
    questionnaire_json = json.dumps(questionnaire or {}, ensure_ascii=False)
    execute(
        "INSERT INTO user_profiles (profile_id, identity_type, identity_label, questionnaire) VALUES (?, ?, ?, ?)",
        (profile_id, identity_type, identity_label, questionnaire_json),
    )
    return get_profile(profile_id)


def get_profile(profile_id: str) -> Optional[dict]:
    """获取用户画像。"""
    row = query_one(
        "SELECT * FROM user_profiles WHERE profile_id = ?",
        (profile_id,),
    )
    if row:
        row["questionnaire"] = json.loads(row.get("questionnaire", "{}"))
    return row


def update_profile(profile_id: str,
                   identity_type: Optional[str] = None,
                   identity_label: Optional[str] = None,
                   questionnaire: Optional[dict] = None) -> Optional[dict]:
    """更新用户画像。"""
    existing = get_profile(profile_id)
    if not existing:
        return None

    fields = []
    values = []
    if identity_type is not None:
        fields.append("identity_type = ?")
        values.append(identity_type)
    if identity_label is not None:
        fields.append("identity_label = ?")
        values.append(identity_label)
    if questionnaire is not None:
        fields.append("questionnaire = ?")
        values.append(json.dumps(questionnaire, ensure_ascii=False))

    if fields:
        fields.append("updated_at = datetime('now')")
        values.append(profile_id)
        execute(
            f"UPDATE user_profiles SET {', '.join(fields)} WHERE profile_id = ?",
            tuple(values),
        )

    return get_profile(profile_id)
