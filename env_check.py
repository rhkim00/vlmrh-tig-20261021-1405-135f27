"""Environment check: GPUs, library versions, HF access, pinned revisions, local cache. Writes outputs/env.json."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys

from .config import output_dir, resolve


def _gpu_info():
    import torch

    gpus = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        gpus.append({"index": i, "name": torch.cuda.get_device_name(i), "total_gb": round(total / 1e9, 1),
                     "free_gb": round(free / 1e9, 1)})
    return gpus


def run_env_check(cfg: dict) -> dict:
    import torch
    from huggingface_hub import HfApi

    from .train import library_versions

    info = {"python": sys.version.split()[0], "platform": platform.platform(), "executable": sys.executable,
            "libraries": library_versions(), "cuda_available": torch.cuda.is_available(), "gpus": _gpu_info()}
    try:
        info["nvidia_smi_driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True,
            text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001
        info["nvidia_smi_driver"] = f"unavailable: {e}"
    import qwen_vl_utils  # noqa: F401  (imported only to confirm it is installed)

    api = HfApi()
    hub = {}
    for name, rid, rtype in (("policy", cfg["models"]["policy"]["id"], "model"),
                             ("judge", cfg["models"]["judge"]["id"], "model"),
                             ("dataset", cfg["data"]["dataset_id"], "dataset")):
        pinned = cfg["models"][name]["revision"] if name != "dataset" else cfg["data"]["revision"]
        try:
            remote = (api.model_info(rid) if rtype == "model" else api.dataset_info(rid)).sha
            hub[name] = {"id": rid, "pinned_revision": pinned, "remote_main": remote,
                         "pinned_is_main": remote == pinned}
        except Exception as e:  # noqa: BLE001
            hub[name] = {"id": rid, "pinned_revision": pinned, "remote_error": repr(e)}
    info["hub"] = hub
    hf_home = resolve(cfg, cfg["project"]["hf_home"])
    info["hf_home"] = str(hf_home)
    info["disk_free_gb_hf_home"] = round(shutil.disk_usage(hf_home if hf_home.exists() else resolve(cfg, ".")).free / 1e9)

    gpus = info["gpus"]
    pd, jd = cfg["devices"]["policy"], cfg["devices"]["judge"]
    if not gpus:
        info["placement"] = "NO GPU: only CPU-level tests (pytest) can run"
    elif pd == jd:
        info["placement"] = (f"policy and judge share {pd} ({gpus[0]['name']}, {gpus[0]['total_gb']} GB); measured "
                             "peak ~26 GB at eval, training peak recorded in run_meta.json")
    else:
        info["placement"] = f"policy on {pd}, judge on {jd}"

    out = output_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "env.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    return info
