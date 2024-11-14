# Set Python 3.10 as the base image and install essential tools
FROM python:3.10-slim

# Install necessary packages
RUN apt-get update && \
    apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama CLI (assuming it is available via curl)
RUN curl -fsSL https://ollama.com/install.sh | sh

# Download and install the Gemma model
RUN ollama pull gemma2:9b || echo "Failed to pull Gemma model"

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt to the container for Python dependency installation
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files to the container
COPY . .

# Set the default command to run the Python script with the Gemma model
CMD ["python3", "bot.py"]
