"""推理链索引 — 存储博主的推演路径结构化数据。

每条推理链是一个结构化 JSON：
  { trigger, chain: [{step, logic_type}], conclusion }

用于检索博主的完整分析框架，而非孤立观点。
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
from chromadb.config import Settings

from src.config import config

# 推理链默认 Schema
REASONING_TYPES = [
    "因果推导",       # A → B → C
    "多空博弈",       # 多方 vs 空方因素
    "历史类比",       # 与历史阶段对比
    "边际变化",       # 关注变化而非绝对值
    "供需推演",       # 供求关系决定价格
    "政策传导",       # 政策 → 市场 → 行为
    "比价关系",       # 不同资产/板块间的价格对标
    "周期判断",       # 行业周期阶段判断
]


class ReasoningChain:
    """单条推理链。"""

    def __init__(
        self, trigger: str, conclusion: str,
        chain: list[dict] | None = None,
        areas: list[str] | None = None,
        logic_tags: list[str] | None = None,
        districts: list[str] | None = None,
    ) -> None:
        self.trigger = trigger
        self.conclusion = conclusion
        self.chain = chain or []
        self.areas = areas or []
        self.logic_tags = logic_tags or []
        self.districts = districts or []

    def add_step(self, step: str, logic_type: str = "") -> None:
        self.chain.append({"step": step, "logic_type": logic_type})

    def to_dict(self) -> dict:
        return {
            "trigger": self.trigger,
            "chain": self.chain,
            "conclusion": self.conclusion,
            "areas": self.areas,
            "logic_tags": self.logic_tags,
        }

    def to_text(self) -> str:
        """转为人可读的文本表示，用于注入提示词。"""
        parts = [f"触发: {self.trigger}"]
        for s in self.chain:
            prefix = f"[{s.get('logic_type', '推演')}]" if s.get("logic_type") else ""
            parts.append(f"  → {prefix} {s['step']}")
        parts.append(f"结论: {self.conclusion}")
        return "\n".join(parts)


class ReasoningIndex:
    """基于 ChromaDB 的推理链索引。"""

    def __init__(self, embed_dim: int = 1024) -> None:
        self.client = chromadb.PersistentClient(
            path=str(config.chroma_db_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection_name = config.reasoning_collection
        self._collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "dim": embed_dim},
        )

    @property
    def collection(self):
        return self._collection

    def add_chains(
        self, chains: list[ReasoningChain], embeddings: list[list[float]],
    ) -> None:
        """添加推理链。"""
        ids = [f"rc_{i:04d}" for i in range(self.collection.count(), self.collection.count() + len(chains))]
        texts = [c.to_text() for c in chains]
        json_strs = [json.dumps(c.to_dict(), ensure_ascii=False) for c in chains]

        metadatas = [
            {
                "trigger": c.trigger[:200],
                "conclusion": c.conclusion[:200],
                "chain_json": js,
                "areas": ",".join(c.areas[:5]) if c.areas else "",
                "logic_tags": ",".join(c.logic_tags[:5]) if c.logic_tags else "",
                "districts": ",".join(c.districts[:3]) if c.districts else "",
            }
            for c, js in zip(chains, json_strs)
        ]

        # 去重（按 trigger + conclusion 前 50 字判重）
        existing_texts = set(self.collection.get(include=["documents"])["documents"])
        new_ids, new_texts, new_metas, new_embs = [], [], [], []
        for i, t in enumerate(texts):
            if t not in existing_texts:
                new_ids.append(ids[i])
                new_texts.append(t)
                new_metas.append(metadatas[i])
                new_embs.append(embeddings[i])

        if new_ids:
            self.collection.add(
                ids=new_ids,
                documents=new_texts,
                metadatas=new_metas,
                embeddings=new_embs,
            )
            print(f"  🧠 新增 {len(new_ids)} 条推理链 (跳过 {len(texts) - len(new_ids)} 条重复)")

    def search(self, query_embedding: list[float], top_k: int = 3,
               where_filter: dict | None = None) -> list[dict]:
        """按语义检索最相关的推理链，支持元数据过滤。

        Args:
            query_embedding: 查询向量
            top_k: 返回数量
            where_filter: ChromaDB where 过滤条件，如 {"areas": {"$contains": "前滩"}}
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 3 if where_filter else top_k,  # 有过滤就扩召回
            include=["documents", "metadatas", "distances"],
            where=where_filter,
        )

        hits = []
        if results["ids"][0]:
            for i in range(len(results["ids"][0])):
                chain_json_str = results["metadatas"][0][i].get("chain_json", "{}")
                try:
                    chain_data = json.loads(chain_json_str)
                except json.JSONDecodeError:
                    chain_data = {}

                hits.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "trigger": results["metadatas"][0][i].get("trigger", ""),
                    "conclusion": results["metadatas"][0][i].get("conclusion", ""),
                    "chain_data": chain_data,
                    "areas": results["metadatas"][0][i].get("areas", ""),
                    "logic_tags": results["metadatas"][0][i].get("logic_tags", ""),
                    "score": 1.0 - results["distances"][0][i],
                })
        return hits[:top_k]

    def update_metadata(self, id_: str, metadata: dict) -> None:
        """更新单条推理链的元数据。"""
        self.collection.update(ids=[id_], metadatas=[metadata])

    def stats(self) -> dict:
        count = self.collection.count()
        return {"count": count}

    def delete(self) -> None:
        self.client.delete_collection(self.collection_name)
        self._collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"},
        )
