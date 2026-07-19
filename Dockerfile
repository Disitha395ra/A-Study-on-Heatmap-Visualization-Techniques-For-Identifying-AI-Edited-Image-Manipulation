FROM python:3.10-slim

# Install system dependencies for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install them
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install huggingface_hub and download the models
RUN pip install --no-cache-dir huggingface_hub
RUN huggingface-cli download Disitha/heatmap-models --local-dir . --repo-type model

# Copy the backend code and necessary modules
COPY backend ./backend
COPY cam ./cam
COPY configs ./configs
COPY metrics ./metrics
COPY models ./models


# Set python path so backend modules can be imported correctly
ENV PYTHONPATH=/app

# Expose the port the app runs on (Hugging Face Spaces requires 7860)
EXPOSE 7860

# Start the application using uvicorn
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
