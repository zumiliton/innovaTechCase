FROM python:3.11-slim AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY models ./models
COPY data ./data
COPY scripts ./scripts

EXPOSE 8000



# OpenAI variant

FROM base AS api

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


# ---------------------------------------------------------
# Local LLM variant
# ---------------------------------------------------------

FROM base AS local

# Local LLM runtime/model will be added here.
# The model is intentionally stored inside the Docker image.

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]