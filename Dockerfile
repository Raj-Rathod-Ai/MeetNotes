FROM python:3.11-slim

WORKDIR /app

# Environment optimizations
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MALLOC_TRIM_THRESHOLD_=100000

# Install ffmpeg and nodejs (required by yt-dlp to solve YouTube JS signature challenges)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install lightweight requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files and static UI
COPY . .

# Expose Render port
EXPOSE 10000

# Run FastAPI server binding dynamically to Render's $PORT
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
