FROM astral/uv:0.8.17-python3.12-bookworm

# Set environment variables

ENV PORT=${PORT}
# Set working directory
WORKDIR /app

# Copy dependency files first to leverage Docker cache
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen
RUN apt update && apt install -y curl tesseract-ocr
# Copy the rest of the application
COPY . .

# Set working directory for Django app
WORKDIR /app/aquillm

# Command to run the application
CMD uv run pytest -s -v \
    --ds=aquillm.settings \
    --reuse-db \
    | tee ../test-results.txt
