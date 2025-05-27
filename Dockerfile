# Use an official Python runtime as a base image
FROM python:3.10-slim

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

# Run the application with Gunicorn
CMD ["gunicorn", "--preload", "-w", "2", "-b", "0.0.0.0:$PORT", "app:app"]