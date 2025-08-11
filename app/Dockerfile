# Base image with Python
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        fluidsynth libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Create tmp directory
RUN mkdir -p /app/tmp

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask runs on
EXPOSE 8080

# Command to run the application
CMD ["gunicorn", "--timeout", "120", "--bind", "0.0.0.0:8080",  "app:app"]