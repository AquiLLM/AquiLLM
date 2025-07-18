FROM ghcr.io/astral-sh/uv:0.6.4-python3.12-bookworm

# Set environment variables

ENV PORT=${PORT}
# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN uv pip install -r requirements.txt --system
RUN apt update && apt install -y curl npm inotify-tools
# Copy the rest of the application
COPY . .


# Set working directory for Django app
WORKDIR /app/aquillm

# Command to run the application
CMD ["sh", "-c", "/app/dev/run.sh"]
