FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py data_loader.py ./
COPY pages/ ./pages/
COPY utils/ ./utils/
COPY api/ ./api/
COPY etl/ ./etl/

# Overridden per-service in docker-compose.yml (uvicorn for the API, streamlit for the UI)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
