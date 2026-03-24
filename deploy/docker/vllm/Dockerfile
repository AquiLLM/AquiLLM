# Pinned: Whisper + bitsandbytes 4-bit fails on v0.18.x (V1 engine, bitsandbytes.py bias shape in encoder profile).
# Bump to :latest (or v0.18+) only after upstream fixes or use fp16 + smaller Whisper without BnB on transcribe.
FROM vllm/vllm-openai:v0.17.1

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir soundfile

COPY --chmod=755 ./deploy/scripts/vllm_start.sh /vllm_start.sh
COPY --chmod=755 ./deploy/scripts/parse_vllm_extra_args.py /parse_vllm_extra_args.py
COPY ./deploy/docker/vllm/chat_templates /templates

EXPOSE 8000
ENTRYPOINT ["/vllm_start.sh"]
