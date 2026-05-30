"""实时搜索引擎客户端 — 对接 AnySearch API，获取最新市场数据。

在静态 RAG 知识库之外，补充实时行情、挂牌数据、政策动态等。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests

from src.config import config

SKILL_ENV_PATH = Path.home() / ".claude" / "skills" / "anysearch" / ".env"
ANYSEARCH_ENDPOINT = "https://api.anysearch.com/mcp"


def _load_anysearch_key() -> str:
    """获取 AnySearch API key。

    优先级：项目 config > 项目 .env > skill 目录 .env
    """
    key = config.anysearch_api_key.strip()
    if key:
        return key

    # 项目 .env（已由 python-dotenv 加载到 os.environ）
    key = os.environ.get("ANYSEARCH_API_KEY", "").strip()
    if key:
        return key

    # skill 目录 .env
    if SKILL_ENV_PATH.exists():
        with open(SKILL_ENV_PATH, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line.startswith("ANYSEARCH_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip("\"'")
                    if key:
                        return key
    return ""


_client: AnySearchClient | None = None


def web_search() -> AnySearchClient:
    """获取 WebSearchClient 单例。"""
    global _client
    if _client is None:
        _client = AnySearchClient()
    return _client


class SearchResult:
    """单条搜索结果。"""

    def __init__(self, title: str, url: str, snippet: str, score: float = 0.0) -> None:
        self.title = title
        self.url = url
        self.snippet = snippet
        self.score = score

    def to_text(self, idx: int) -> str:
        return f"[实时{idx}] {self.title}\n   来源: {self.url}\n   {self.snippet}"

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "score": self.score,
        }


class AnySearchClient:
    """AnySearch API 客户端，无需子进程，直接 HTTP 调用。"""

    def __init__(self) -> None:
        self.api_key = _load_anysearch_key()
        self._available = bool(self.api_key)

    @property
    def available(self) -> bool:
        return self._available

    def _call(self, tool_name: str, arguments: dict) -> str | None:
        """调用 AnySearch JSON-RPC API。"""
        if not self._available:
            return None

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }

        try:
            resp = requests.post(
                ANYSEARCH_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=15,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException:
            return None

        data = resp.json()
        if "error" in data:
            return None

        result = data.get("result", {})
        content = result.get("content", [])
        for item in content:
            if item.get("type") == "text":
                return item.get("text", "")
        return json.dumps(result, indent=2, ensure_ascii=False) if result else None

    def search(
        self, query: str, max_results: int = 5,
        freshness: str | None = "year",
        domain: str | None = None,
        zone: str | None = None,
    ) -> list[SearchResult]:
        """通用搜索，返回结构化结果列表。"""
        args: dict = {
            "query": query,
            "max_results": max_results,
        }
        if domain:
            args["domain"] = domain
        if zone:
            args["zone"] = zone

        raw = self._call("search", args)
        return self._parse_search_results(raw)

    def search_news(
        self, query: str, max_results: int = 5, freshness: str = "month",
        domain: str | None = None, zone: str | None = None,
    ) -> list[SearchResult]:
        """新闻搜索。"""
        args: dict = {
            "query": query,
            "max_results": max_results,
            "content_types": ["news"],
            "freshness": freshness,
        }
        if domain:
            args["domain"] = domain
        if zone:
            args["zone"] = zone

        raw = self._call("search", args)
        return self._parse_search_results(raw)

    def extract(self, url: str) -> str | None:
        """提取网页内容为 Markdown。"""
        return self._call("extract", {"url": url})

    @staticmethod
    def _parse_search_results(raw: str | None) -> list[SearchResult]:
        """解析 AnySearch 的 Markdown 格式搜索结果。"""
        if not raw:
            return []

        results: list[SearchResult] = []
        current_title = ""
        current_url = ""
        current_snippet: list[str] = []

        for line in raw.split("\n"):
            # 匹配结果项: "### N. Title"
            if line.startswith("### "):
                # 保存上一条
                if current_title:
                    results.append(SearchResult(
                        title=current_title,
                        url=current_url,
                        snippet=" ".join(current_snippet).strip(),
                    ))
                current_title = line[4:].strip()
                # 去掉序号前缀 "1. "
                if ". " in current_title:
                    current_title = current_title.split(". ", 1)[-1]
                current_url = ""
                current_snippet = []
            elif line.strip().startswith("- **URL**:"):
                current_url = line.split("**:", 1)[-1].strip()
            elif line.strip() and current_title:
                current_snippet.append(line.strip())

        if current_title:
            results.append(SearchResult(
                title=current_title,
                url=current_url,
                snippet=" ".join(current_snippet).strip(),
            ))

        return results


def format_web_results(results: list[SearchResult]) -> str:
    """将搜索结果格式化为 prompt 可用的文本。"""
    if not results:
        return ""

    parts = ["## 实时行情数据（来自网络搜索，可能存在时效性差异）"]
    for i, r in enumerate(results, 1):
        parts.append(r.to_text(i))
    parts.append("")
    return "\n\n".join(parts)
