FROM python:3.11-slim

WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/

# Install CPU-only torch first (avoids pulling in 2GB+ of CUDA libraries on GPU-less servers)
RUN pip install --no-cache-dir --default-timeout=300 -i https://pypi.tuna.tsinghua.edu.cn/simple torch --extra-index-url https://download.pytorch.org/whl/cpu
# Install serving dependencies
RUN pip install --no-cache-dir --default-timeout=300 -i https://pypi.tuna.tsinghua.edu.cn/simple -e ".[serve]"

# Pre-download embedding and reranker models so they're cached in the image
# and work without network access at runtime (bge-m3 ~2.3GB, bge-reranker-v2-m3 ~2.1GB)
RUN for i in 1 2 3; do \
      HF_ENDPOINT=https://hf-mirror.com \
        hf download BAAI/bge-m3 && break; \
      echo "bge-m3 attempt $i failed, retrying..."; \
      sleep 10; \
    done && \
    for i in 1 2 3; do \
      HF_ENDPOINT=https://hf-mirror.com \
        hf download BAAI/bge-reranker-v2-m3 && break; \
      echo "bge-reranker attempt $i failed, retrying..."; \
      sleep 10; \
    done

# Copy ChromaDB knowledge base
COPY data/ ./data/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501')" || exit 1

CMD ["streamlit", "run", "src/app/ui.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
