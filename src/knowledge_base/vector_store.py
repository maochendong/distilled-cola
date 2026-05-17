"""知识索引 — 基于 ChromaDB 存储带标注的上海房产知识片段。"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
from chromadb.config import Settings

from src.config import config


class KnowledgeIndex:
    """存储带房产领域标注的知识片段。"""

    def __init__(self, embed_dim: int = 1024) -> None:
        self.client = chromadb.PersistentClient(
            path=str(config.chroma_db_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection_name = config.knowledge_collection
        self._collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "dim": embed_dim},
        )

    @property
    def collection(self):
        return self._collection

    def add_blocks(self, blocks: list[dict], embeddings: list[list[float]]) -> None:
        """将带标注的知识块写入索引。"""
        ids = [b["id"] for b in blocks]
        texts = [b["text"] for b in blocks]

        # 提取标注元数据供过滤查询
        metadatas = []
        for b in blocks:
            ann = b.get("annotation", {})
            meta = {
                "source": b.get("source", ""),
                "title": b.get("title", "")[:200],
                "chunk_index": b.get("chunk_index", 0),
                "length": b.get("length", 0),
                "is_complete": str(b.get("is_complete_analysis", False)),
            }
            if ann.get("logic_tags"):
                meta["logic_tags"] = ",".join(ann["logic_tags"][:5])
            if ann.get("suggestion_tags"):
                meta["suggestion_tags"] = ",".join(ann["suggestion_tags"][:3])
            if ann.get("entity_tags", {}).get("areas"):
                meta["areas"] = ",".join(ann["entity_tags"]["areas"][:3])
            if ann.get("entity_tags", {}).get("districts"):
                meta["districts"] = ",".join(ann["entity_tags"]["districts"][:3])
            metadatas.append(meta)

        # 去重
        existing = set(self.collection.get(ids=ids, include=[])["ids"])
        new_ids, new_texts, new_metas, new_embs = [], [], [], []
        for i, _id in enumerate(ids):
            if _id not in existing:
                new_ids.append(_id)
                new_texts.append(texts[i])
                new_metas.append(metadatas[i])
                new_embs.append(embeddings[i])

        if new_ids:
            self.collection.add(
                ids=new_ids,
                documents=new_texts,
                metadatas=new_metas,
                embeddings=new_embs,
            )
            print(f"  📥 新增 {len(new_ids)} 条知识 (跳过 {len(ids) - len(new_ids)} 条重复)")

    def search(
        self, query_embedding: list[float], top_k: int | None = None,
        where_filter: dict | None = None,
    ) -> list[dict]:
        """向量检索，支持按标注标签过滤。"""
        k = top_k or config.top_k
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k * 2 if where_filter else k,
            include=["documents", "metadatas", "distances"],
            where=where_filter,
        )

        hits = []
        if results["ids"][0]:
            for i, _id in enumerate(results["ids"][0]):
                hits.append({
                    "id": _id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1.0 - results["distances"][0][i],
                })
        return hits[:k]

    def search_by_areas(self, query_embedding: list[float], areas: list[str], top_k: int = 3) -> list[dict]:
        """按板块过滤检索。"""
        area_filter = {"areas": {"$contains": areas[0]}} if areas else None
        return self.search(query_embedding, top_k=top_k, where_filter=area_filter)

    def stats(self) -> dict:
        """统计信息。"""
        count = self.collection.count()
        if count == 0:
            return {"count": 0, "sources": [], "tags": []}
        all_meta = self.collection.get(include=["metadatas"])
        sources = list({m.get("source", "") for m in all_meta["metadatas"] if m.get("source")})
        all_tags: set[str] = set()
        for m in all_meta["metadatas"]:
            if m.get("logic_tags"):
                all_tags.update(m["logic_tags"].split(","))
        return {"count": count, "sources": sources, "logic_tags": list(all_tags)}

    def export_to_json(self, path: Path | None = None) -> Path:
        """导出知识库为 JSON。"""
        out = path or config.processed_dir / "knowledge_index_export.json"
        all_data = self.collection.get(include=["documents", "metadatas"])
        export = [
            {"id": all_data["ids"][i], "text": all_data["documents"][i], "metadata": all_data["metadatas"][i]}
            for i in range(len(all_data["ids"]))
        ]
        out.write_text(json.dumps(export, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    def delete(self) -> None:
        """清空索引。"""
        self.client.delete_collection(self.collection_name)
        self._collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"},
        )
