# Base image with Python
FROM python:3.12-slim

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