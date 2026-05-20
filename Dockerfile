FROM python:3.11-slim

WORKDIR /app

# ============================================================
# Deps layer (rebuilds only when pyproject.toml changes)
# ============================================================
COPY pyproject.toml ./

# Minimal package so pip install -e . can resolve
RUN mkdir -p src/app src/collector src/rag src/knowledge_base && \
    touch src/__init__.py src/app/__init__.py src/collector/__init__.py \
          src/rag/__init__.py src/knowledge_base/__init__.py

# --mount=type=cache persists pip's download cache across rebuilds
# (BuildKit cache mount, NOT included in final image)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=300 -i https://pypi.tuna.tsinghua.edu.cn/simple \
      "sympy>=1.13.3" "networkx>=2.5.1" "jinja2>=3.0" "fsspec>=0.8.5" \
      "mpmath>=1.1.0,<1.4" "MarkupSafe>=2.0" "typing-extensions>=4.10.0"

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=300 --no-deps torch \
      --index-url https://download.pytorch.org/whl/cpu

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=300 -i https://pypi.tuna.tsinghua.edu.cn/simple pillow
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=300 --no-deps torchvision \
      --index-url https://download.pytorch.org/whl/cpu

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=300 -i https://pypi.tuna.tsinghua.edu.cn/simple -e ".[serve]"

# ============================================================
# Source layer (rebuilds when src/ or data/ changes, but pip
# layers above are cached, so rebuilds are fast)
# ============================================================
COPY src/ ./src/
COPY data/ ./data/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501')" || exit 1

CMD ["streamlit", "run", "src/app/ui.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
