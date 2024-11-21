# Use the official Python 3.10 base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /workspace

# Install system dependencies required by data science tools (e.g., numpy, pandas, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    make \
    libatlas-base-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    && apt-get clean

# Copy the rest of the project files
COPY . .

# Set default command to start a bash shell
CMD ["bash"]