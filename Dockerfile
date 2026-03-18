FROM astral/uv:0.8.17-python3.12-bookworm

ENV PORT=${PORT}
ENV UV_PROJECT_ENVIRONMENT=/usr/local

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project
RUN apt update && apt install -y curl npm inotify-tools

COPY . .

WORKDIR /app/aquillm

CMD ["sh", "-c", "/app/dev/run.sh"]
