# Qwen fixed chat template (vendored)

- **Source:** [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) (v19, Apache-2.0)
- **File:** `qwen_fixed_chat_template.jinja` → mounted in vLLM images at `/templates/qwen_fixed_chat_template.jinja`
- **Enable:** `--chat-template /templates/qwen_fixed_chat_template.jinja` in `VLLM_EXTRA_ARGS` (see `.env.example`)

Complements [Genesis](https://github.com/Sandermage/genesis-vllm-patches) runtime patches; does not replace them.

To refresh from upstream:

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('froggeric/Qwen-Fixed-Chat-Templates','chat_template.jinja',local_dir='deploy/docker/vllm/chat_templates', force_download=True)"
# Then rename chat_template.jinja -> qwen_fixed_chat_template.jinja and remove .cache if created.
```
