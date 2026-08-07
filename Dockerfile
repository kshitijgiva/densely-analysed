FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py chatbot.py ./
COPY db/ ./db/

EXPOSE 8080
# Railway injects PORT; default 8080 for local docker compose.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080}"]
