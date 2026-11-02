FROM ubuntu:22.04 AS mimalloc-builder
COPY deploy/docker/mimalloc/build.sh deploy/docker/mimalloc/verify.c /build/
RUN sh /build/build.sh

FROM astral/uv:0.8.17-python3.12-bookworm

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
ENV HF_HOME=/root/.cache/huggingface
ENV KG_GLINER2_CACHE_DIR=/root/.cache/huggingface
ENV KG_GLINER2_DEVICE=cpu
ENV KG_GLINER2_LOCAL_FILES_ONLY=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV KG_EXTRACTION_QUEUE=knowledge-graph-extraction

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_INSTALLS=1 uv sync --frozen --no-dev --extra knowledge-graph-local --no-cache
RUN /opt/venv/bin/python -c "from importlib.metadata import version; import neo4j._codec.packstream.v1 as codec; \
    assert version('neo4j') == '5.28.4' and version('neo4j-rust-ext') == '5.28.4.0', 'KG runtime requires neo4j==5.28.4 and neo4j-rust-ext==5.28.4.0'; \
    assert codec._rust_pack is not None and codec._rust_unpack is not None, 'KG runtime requires compiled PackStream encoder and decoder'"
RUN /opt/venv/bin/python -c "import torch; assert torch.version.cuda is None, torch.version.cuda"

# Projection workers start without public network access; retain tokenizer data
# outside /app so a deployment bind mount cannot hide the cache.
ENV TIKTOKEN_CACHE_DIR=/opt/aquillm-tokenizers
RUN /opt/venv/bin/python -c "import tiktoken; tiktoken.get_encoding('o200k_base'); tiktoken.get_encoding('cl100k_base')"

COPY . .

WORKDIR /app/aquillm

COPY --from=mimalloc-builder /opt/mimalloc/ /opt/mimalloc/
COPY --chmod=755 deploy/scripts/with_mimalloc.sh /usr/local/bin/aquillm-allocator
ENV AQUILLM_ALLOCATOR=mimalloc
ENTRYPOINT ["/usr/local/bin/aquillm-allocator"]
CMD ["/opt/venv/bin/python", "-c", "from lib.knowledge_graph.config import run_extraction_worker; run_extraction_worker()"]
