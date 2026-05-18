FROM python:3.11-slim

WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/

# Install serving dependencies only (no sentence-transformers/torch)
RUN pip install --no-cache-dir --default-timeout=300 -i https://pypi.tuna.tsinghua.edu.cn/simple -e ".[serve]"

# Copy ChromaDB knowledge base
COPY data/ ./data/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501')" || exit 1

CMD ["streamlit", "run", "src/app/ui.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
