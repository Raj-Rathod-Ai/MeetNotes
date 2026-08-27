FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (ffmpeg, git, build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files and static UI assets
COPY . .

# Expose port
EXPOSE 8000

# Start high-performance FastAPI server with dynamic Render $PORT support
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
