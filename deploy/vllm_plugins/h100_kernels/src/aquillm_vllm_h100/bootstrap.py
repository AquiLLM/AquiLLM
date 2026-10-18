"""Ordered, idempotent adapter registration after Genesis has applied patches."""
import json
import logging
import os

log = logging.getLogger("aquillm.h100")
_installed = False


def settings(env):
    result = dict(mtp=env.get("AQUILLM_H100_MTP_KERNEL", "baseline"),
                  split=env.get("AQUILLM_H100_SPLIT_POLICY", "baseline"),
                  prefill=env.get("AQUILLM_H100_PREFILL", "0"),
                  gdn=env.get("AQUILLM_H100_GDN", "baseline"),
                  runtime_profile=env.get("AQUILLM_H100_RUNTIME_PROFILE", "baseline"),
                  profile=env.get("AQUILLM_H100_PROFILE"))
    if result["mtp"] not in ("baseline", "fused"):
        raise ValueError("unknown MTP kernel setting")
    if result["split"] not in ("baseline", "adaptive"):
        raise ValueError("unknown split policy")
    if result["prefill"] not in ("0", "1"):
        raise ValueError("unknown prefill setting")
    if result["gdn"] not in ("baseline", "flashinfer", "native-fp16"):
        raise ValueError("unknown GDN adapter setting")
    if ((result["gdn"] == "native-fp16")
            != (result["runtime_profile"] == "native-gdn-baseline")):
        raise ValueError("native-fp16 requires runtime profile native-gdn-baseline together")
    if result["gdn"] == "flashinfer" and result["runtime_profile"] != "flashinfer-0.6.18":
        raise ValueError("FlashInfer GDN requires candidate runtime flashinfer-0.6.18")
    return result


def install(env=None):
    global _installed
    config = settings(os.environ if env is None else env)
    if (config["mtp"] == config["split"] == config["gdn"] == "baseline"
            and config["prefill"] == "0" and config["runtime_profile"] == "baseline"):
        return {"status": "disabled"}
    if _installed:
        return {"status": "already_installed"}
    try:
        from .compatibility import verify_runtime
        verify_runtime()
        from .adapters import install_adapters
        result = install_adapters(config)
    except Exception as error:
        result = {"status": "inactive", "reason": str(error)}
        log.error("AQUILLM_H100 %s", json.dumps(result))
        return result
    _installed = True
    log.warning("AQUILLM_H100 %s", json.dumps(result))
    return result
