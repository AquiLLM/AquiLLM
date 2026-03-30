FROM astral/uv:0.8.17-python3.12-bookworm

# Set environment variables

ENV PORT=${PORT}
# Set working directory
WORKDIR /app

# Copy dependency files first to leverage Docker cache
COPY pyproject.toml uv.lock ./
RUN UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_INSTALLS=1 uv sync --frozen --no-dev --no-cache
RUN apt update && apt install -y curl npm inotify-tools tesseract-ocr \
    libreoffice-writer-nogui libreoffice-calc-nogui libreoffice-impress-nogui
# Copy the rest of the application
COPY . .


# Set working directory for Django app
WORKDIR /app/aquillm

# Command to run the application
CMD ["sh", "/app/deploy/scripts/dev/run.sh"]
