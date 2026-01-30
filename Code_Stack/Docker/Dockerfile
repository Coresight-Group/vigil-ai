FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY app.py .
COPY config.py .
COPY utils.py .

# NOTE: Do NOT copy .env files into Docker image
# Environment variables should be passed via docker-compose or runtime secrets

# ============================================================================
# FRONTEND FILES
# These should be served from a proper static file location
# Update paths based on your deployment structure
# ============================================================================
# Note: Frontend files are typically served by Nginx in production
# Copy them only if Flask is serving static files directly

# Create static directory for frontend
RUN mkdir -p /app/static

# Frontend files should be mounted or copied from FrontEnd directory
# COPY ../FrontEnd/Interface/index.html /app/static/
# COPY ../FrontEnd/Interface/styles.css /app/static/
# COPY ../FrontEnd/Interface/script.js /app/static/
# ============================================================================

# Expose ports
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Run Flask backend
CMD ["python", "app.py"]