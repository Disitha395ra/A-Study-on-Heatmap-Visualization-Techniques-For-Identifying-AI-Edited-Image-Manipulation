FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install them
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install huggingface_hub and download the models
RUN pip install --no-cache-dir huggingface_hub
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Disitha/heatmap-models', local_dir='.', repo_type='model')"

# Copy the backend code and necessary modules
COPY backend ./backend
COPY cam ./cam
COPY configs ./configs
COPY metrics ./metrics
COPY models ./models


# Set python path so backend modules can be imported correctly
ENV PYTHONPATH=/app

# Expose the port the app runs on (Hugging Face Spaces requires 7860, Render uses PORT)
EXPOSE 7860

# Start the application using uvicorn
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
