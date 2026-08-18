FROM astral/uv:0.8.17-python3.12-bookworm

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
ENV HF_HOME=/root/.cache/huggingface
ENV KG_GLINER2_DEVICE=cpu

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_INSTALLS=1 uv sync --frozen --no-dev --extra knowledge-graph-local --no-cache
RUN /opt/venv/bin/python -c "import torch; assert torch.version.cuda is None, torch.version.cuda"

COPY . .

WORKDIR /app/aquillm

CMD ["celery", "-A", "aquillm", "worker", "--loglevel=info", "--queues=knowledge-graph-extraction", "--concurrency=1", "--prefetch-multiplier=1", "--hostname=worker-knowledge-graph@%h"]
