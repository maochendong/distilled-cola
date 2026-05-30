"""CLI 入口 — 上海房产分析专家命令行交互。"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import config
from src.rag.pipeline import RAGPipeline

app = typer.Typer(
    name="cola",
    help="蒸馏小可乐 — 上海房产分析专家",
    no_args_is_help=True,
)
console = Console()


@app.command()
def ask(
    query: str = typer.Argument(..., help="你的问题，如「800万预算前滩vs大宁怎么选？」"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="检索的知识块数量"),
) -> None:
    """向蒸馏后的上海房产专家提问。"""
    pipe = RAGPipeline()
    with console.status("🔍 正在检索博主知识库 + 实时行情并生成四步分析..."):
        result = pipe.ask(query, top_k=top_k)

    console.print()
    console.print(Panel(
        result["answer"],
        title=f"💡 分析 (置信度: {result['confidence']:.1%})",
        border_style="green",
    ))
    console.print()

    if result.get("reasoning_chains_used", 0) > 0:
        console.print(f"🧠 引用了 {result['reasoning_chains_used']} 条推理链")
    if result.get("web_search_used", False):
        console.print("🌐 参考了实时网络搜索数据")

    if result["sources"]:
        table = Table(title="📚 参考来源", show_header=True)
        table.add_column("来源", style="cyan")
        table.add_column("相关度", justify="right")
        table.add_column("片段")
        for s in result["sources"]:
            table.add_row(s.get("source", ""), str(s.get("score", "")), s.get("snippet", ""))
        console.print(table)


@app.command()
def build(
    input_file: str = typer.Option("data/processed/knowledge_blocks.json", "--input", "-i",
                                   help="标注后的知识块 JSON 路径"),
) -> None:
    """构建/更新知识库（标注数据 → 向量化 → 存入 ChromaDB + BM25）。"""
    from src.knowledge_base.embedder import Embedder
    from src.knowledge_base.hybrid_retriever import HybridRetriever
    from src.knowledge_base.vector_store import KnowledgeIndex

    input_path = Path(input_file)
    if not input_path.exists():
        console.print(f"[red]❌ 文件不存在: {input_file}")
        console.print("请先完成标注，或使用 `cola import` 导入视频[/red]")
        raise typer.Exit(1)

    console.print("[bold]🔨 构建上海房产知识库...[/bold]")
    blocks = json.loads(input_path.read_text(encoding="utf-8"))
    console.print(f"📦 共 {len(blocks)} 条知识块")

    console.print("🔮 生成向量嵌入...")
    embedder = Embedder()
    texts = [b["text"] for b in blocks]
    embeddings = embedder.embed(texts)
    console.print("   ✅ 嵌入完成")

    console.print("💾 写入知识索引 (向量)...")
    kb = KnowledgeIndex()
    kb.add_blocks(blocks, embeddings)

    console.print("🔍 构建 BM25 倒排索引...")
    retriever = HybridRetriever()
    retriever.build_bm25_index()

    stats = kb.stats()
    console.print(f"[green]✅ 知识库构建完成！[/green]")
    console.print(f"   📊 总知识块: {stats['count']}")
    console.print(f"   📹 来源视频: {len(stats['sources'])}")
    console.print(f"   🏷️  逻辑标签: {len(stats.get('logic_tags', []))}")


@app.command()
def import_video(
    source: str = typer.Argument(..., help="YouTube 视频 ID 或本地视频文件路径"),
    title: str = typer.Option("", "--title", "-t", help="视频标题"),
    platform: str = typer.Option("youtube", "--platform", "-p",
                                 help="数据源平台: youtube | local (微信视频号/小红书)"),
    enable_ocr: bool = typer.Option(True, "--ocr/--no-ocr", help="启用画面文字识别"),
) -> None:
    """导入视频：转录 → 画面OCR → 切分 → 标注 → 入库。

    支持 YouTube 视频 ID 和本地视频文件（微信视频号/小红书下载后）。
    启用 OCR 后会自动提取视频画面中的文字（数据表格、政策截图等）。
    """
    from src.collector.pipeline import process_local_video, process_youtube_video

    if platform == "youtube" and not source.startswith("/"):
        console.print(f"[bold]🎬 导入 YouTube: {source}[/bold]")
        annotated = process_youtube_video(source, title=title)
    else:
        # 本地文件（微信视频号/小红书）
        video_path = source
        if not Path(video_path).exists():
            console.print(f"[red]❌ 文件不存在: {video_path}[/red]")
            raise typer.Exit(1)
        console.print(f"[bold]🎬 导入本地视频: {Path(video_path).name}[/bold]")
        annotated = process_local_video(video_path, title=title, enable_ocr=enable_ocr)

    index_and_finish(annotated)


@app.command()
def import_image(
    image_path: str = typer.Argument(..., help="图片文件路径（截图/数据表/户型图等）"),
    title: str = typer.Option("", "--title", "-t", help="图片标题或描述"),
) -> None:
    """导入图片：OCR → 结构化提取 → 入库。

    适用于博主分享的数据截图、政策文件、户型图、价格表等。
    """
    from src.collector.pipeline import process_local_image

    if not Path(image_path).exists():
        console.print(f"[red]❌ 文件不存在: {image_path}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]🖼️ 导入图片: {Path(image_path).name}[/bold]")
    annotated = process_local_image(image_path, title=title)

    index_and_finish(annotated)


def index_and_finish(annotated: list[dict]) -> None:
    """将标注后的知识块入库并构建检索索引。"""
    from src.knowledge_base.embedder import Embedder
    from src.knowledge_base.hybrid_retriever import HybridRetriever
    from src.knowledge_base.vector_store import KnowledgeIndex

    console.print("🔮 生成向量嵌入...")
    embedder = Embedder()
    texts = [b["text"] for b in annotated]
    embeddings = embedder.embed(texts)

    console.print("💾 写入知识索引...")
    kb = KnowledgeIndex()
    kb.add_blocks(annotated, embeddings)

    console.print("🔍 重建 BM25 索引...")
    retriever = HybridRetriever()
    retriever.build_bm25_index()

    console.print(f"[green]✅ 导入完成！{len(annotated)} 段标注知识已入库[/green]")


@app.command()
def stats() -> None:
    """知识库统计。"""
    from src.knowledge_base.vector_store import KnowledgeIndex
    from src.knowledge_base.reasoning_index import ReasoningIndex

    kb = KnowledgeIndex()
    ri = ReasoningIndex()
    ks = kb.stats()
    rc = ri.stats()

    console.print("[bold]📊 知识库状态[/bold]")
    console.print(f"  知识块: {ks.get('count', 0)}")
    console.print(f"  推理链: {rc.get('count', 0)}")
    console.print(f"  来源: {', '.join(ks.get('sources', ['无'])) if ks.get('sources') else '无'}")
    if ks.get("logic_tags"):
        console.print(f"  逻辑标签: {', '.join(ks['logic_tags'][:10])}")


@app.command()
def streamlit(
    port: int = 8501,
) -> None:
    """启动 Streamlit 交互界面。"""
    import subprocess
    import sys
    ui_path = Path(__file__).parent / "ui.py"
    console.print(f"[bold]🌐 启动 Streamlit: http://localhost:{port}[/bold]")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(ui_path),
        "--server.port", str(port),
        "--server.headless", "true",
    ])


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    """启动 API 服务 (FastAPI)。"""
    import uvicorn
    console.print(f"[bold]🌐 API: http://{host}:{port}[/bold]")
    console.print(f"[bold]📖 Docs: http://{host}:{port}/docs[/bold]")
    uvicorn.run("src.app.api:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    app()
