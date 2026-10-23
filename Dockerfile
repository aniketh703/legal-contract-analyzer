FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (e.g., Tesseract OCR for PDF extraction if used)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libpoppler-cpp-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download HuggingFace models during build to cache them
# We can run a small python snippet to pre-download the models
RUN python -c "from transformers import AutoTokenizer, AutoModel, pipeline; \
AutoTokenizer.from_pretrained('law-ai/InLegalBERT'); \
AutoModel.from_pretrained('law-ai/InLegalBERT'); \
pipeline('summarization', model='sshleifer/distilbart-cnn-12-6', device=-1);"

# Copy the rest of the application
COPY . .

# Set environment variables
ENV FLASK_APP=app.main:app
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

EXPOSE 7860

# Run gunicorn for production. $PORT is honored for platforms that assign one
# (e.g. Hugging Face Spaces); defaults to 7860 (the Spaces Docker SDK convention).
CMD ["sh", "-c", "gunicorn --workers 2 --bind 0.0.0.0:${PORT:-7860} --timeout 120 app.main:app"]
