# Use an official Python runtime as a base image
FROM python:3.10-slim

# Avoid interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics

# Install system packages needed by OpenCV
RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary folders (in case your code depends on them)
RUN mkdir -p uploads frames output_images

# Run the application with Gunicorn
# CMD ["gunicorn", "--preload", "-w", "2", "-b", "0.0.0.0:8080", "app:app"]
# CMD exec gunicorn --bind :$PORT --workers 4 --threads 8 --timeout 120 app:app
CMD ["sh", "-c", "exec gunicorn --bind :$PORT --workers 4 --threads 8 --timeout 120 app:app"]
