# backend/Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# Set the path for transformers cache (optional, helps manage cache location)
ENV TRANSFORMERS_CACHE=/app/.cache/transformers
ENV HF_HOME=/app/.cache/huggingface

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (if any are needed later, e.g., for specific libraries)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# Create a non-root user and group
RUN addgroup --system app && adduser --system --group app

# Create cache directories and set permissions
RUN mkdir -p /app/.cache/transformers /app/.cache/huggingface && chown -R app:app /app/.cache

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir reduces image size
# Consider adding --extra-index-url for torch if needed (e.g., for specific CUDA versions)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the local backend app directory contents into the container at /app/app
COPY ./app /app/app

# Copy the saved model directory into the container at /app/saved_model
# IMPORTANT: This assumes the 'saved_model' directory exists in the build context (backend folder)
# and contains the necessary LLM files. These files are NOT included in this setup.
COPY ./saved_model /app/saved_model

# Change ownership of the app directory to the non-root user
RUN chown -R app:app /app

# Switch to the non-root user
USER app

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define the command to run the app using uvicorn
# Use --host 0.0.0.0 to make it accessible from outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

