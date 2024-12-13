# Use Python 3.9 slim image as the base
FROM python:3.10.16-slim

# Set environment variables to avoid bytecode creation and buffer issues
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /sentiment_analysis/sentiment_analysis_app/app/

# # Copy all the necessary files from your local folder to the container
COPY . /sentiment_analysis/sentiment_analysis_app/

# Install system dependencies (gcc for compiling certain libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
#RUN cd app && pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Expose the port Flask app will run on (default: 5000)
EXPOSE 5000

# Define the command to run the Flask app
CMD ["python", "app.py"]