FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY api.py chatbot.py ./
COPY db/ ./db/

# CLIP text encoder + visual-search Chroma client for chatbot.py's search_visual
# tool - same modules the analytics pipeline uses for ingestion (visual_embeddings.py).
# Keep packages importable: need __init__.py placeholders and services/ as a package.
RUN mkdir -p analytics_service/src/services \
    && touch analytics_service/__init__.py \
             analytics_service/src/__init__.py \
             analytics_service/src/services/__init__.py
COPY analytics_service/src/config.py analytics_service/src/visual_embeddings.py ./analytics_service/src/
COPY analytics_service/src/services/visual_search.py ./analytics_service/src/services/

EXPOSE 8080
# Railway injects PORT; default 8080 for local docker compose.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080}"]
