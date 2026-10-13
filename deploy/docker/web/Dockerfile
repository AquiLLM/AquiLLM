FROM ubuntu:22.04 AS mimalloc-builder
COPY deploy/docker/mimalloc/build.sh deploy/docker/mimalloc/verify.c /build/
RUN sh /build/build.sh

FROM astral/uv:0.8.17-python3.12-bookworm

# Set environment variables

ENV PORT=${PORT}
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
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
COPY --from=mimalloc-builder /opt/mimalloc/ /opt/mimalloc/
COPY --chmod=755 deploy/scripts/with_mimalloc.sh /usr/local/bin/aquillm-allocator
ENV AQUILLM_ALLOCATOR=mimalloc
ENTRYPOINT ["/usr/local/bin/aquillm-allocator"]
CMD ["sh", "/app/deploy/scripts/dev/run.sh"]
