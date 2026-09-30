FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf PYTHONPATH=/app
WORKDIR /app

# CPU-only PyTorch (small image, no GPU needed), then the pinned requirements
RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so the container runs offline
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY . .
EXPOSE 8000

# One command: run the automated replay benchmark, then serve the API
CMD ["sh", "-c", "python -m eval.run_eval && uvicorn app.api:app --host 0.0.0.0 --port 8000"]
