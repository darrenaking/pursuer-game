FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY server/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY server/ ./server/

# Copy frontend
COPY frontend/ ./frontend/

# Create weights directory
RUN mkdir -p server/weights

WORKDIR /app/server

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8000"]
