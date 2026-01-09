# slim
FROM python:3.12-slim as builder

WORKDIR /app

# Installing build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install to /opt/venv instead of user directory
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# final image
FROM python:3.12-slim

WORKDIR /app

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY . .

# Create non-root user
RUN useradd -m delivery_user && \
    chown -R delivery_user:delivery_user /app
USER delivery_user

# FastAPI default port is 8000
EXPOSE 8000

# Running FastAPI with uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]