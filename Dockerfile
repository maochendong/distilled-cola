FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir "setuptools>=68.0" && \
    pip install --no-cache-dir \
        openai>=1.0.0 \
        chromadb>=0.5.0 \
        fastapi>=0.109.0 \
        uvicorn[standard]>=0.27.0 \
        pydantic>=2.0.0 \
        python-dotenv>=1.0.0 \
        rich>=13.0.0 \
        tiktoken>=0.6.0 \
        numpy>=1.24.0 \
        rank-bm25>=0.2.0 \
        jieba>=0.42.0 \
        requests>=2.28.0

# Copy application code
COPY src/ src/
COPY .env .env

# Create data directories for volume mounts
RUN mkdir -p data/embeddings

EXPOSE 8080

CMD ["uvicorn", "src.app.api:app", "--host", "0.0.0.0", "--port", "8080"]
