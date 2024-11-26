# Use Python 3.11 as base image (latest version compatible with TensorFlow)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for TA-Lib and other packages
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xvzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib-0.4.0-src.tar.gz ta-lib/

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create directory for logs
RUN mkdir -p logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC
ENV PYTHONPATH=/app

# Create a non-root user
RUN useradd -m -u 1000 trader
RUN chown -R trader:trader /app

# Switch to non-root user
USER trader

# Command to run the trading bot
CMD ["python", "-m", "bots.future_trading"]
