# v0.21.0 brings the Qwen3.6/MTP/TurboQuant serving path needed by the
# current remote profile. Keep libsndfile/soundfile for the Whisper sidecar.
FROM vllm/vllm-openai:v0.21.0

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
