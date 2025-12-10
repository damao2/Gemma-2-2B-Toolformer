# fuse_delta_transcoders.py
# Fuse base npz and delta pt into JumpReLU.npz and generate the local config.yaml (hook_mlp_in -> hook_mlp_out)

import os, re, math, json
import numpy as np
import torch
import torch.nn as nn
from typing import Optional
try:
    from huggingface_hub import hf_hub_download, HfApi  # optional
except Exception:
    hf_hub_download = None
    HfApi = None

# 1) Configuration area: modify as needed
DELTA_ROOT = "/Volumes/Untitled/Downloads/train_4/dnew_32_pctl_0.5"  # your delta directory
OUT_ROOT   = "/Volumes/Untitled/Downloads/fused_delta_32_pctl_0.5"   # output directory
# Local base npz root directory (can be empty; if empty and online download is allowed, will fetch from HF)
LOCAL_BASE_ROOT = ""  # e.g., "/Users/damao/local_base_transcoders/width_16k"
# Base repository used during training (only used if local is missing and online download is allowed)
BASE_REPO_ID = "google/gemma-scope-2b-pt-transcoders"
WIDTH_DIR = "width_16k"
# Which base npz to select per layer (percentile strategy); the percentile you used during training
SELECTION_PERCENTILE = 0.5
# Whether to allow online download of base (set False for fully offline and prepare LOCAL_BASE_ROOT)
ALLOW_ONLINE_DOWNLOAD = True

def load_base_npz_path(layer_idx: int) -> Optional[str]:
    # Try local first
    if LOCAL_BASE_ROOT:
        # Try to match average_l0_*/params.npz locally, select one by percentile
        cand = []
        for root, _, files in os.walk(os.path.join(LOCAL_BASE_ROOT, f"layer_{layer_idx}", WIDTH_DIR)):
            for f in files:
                if f.endswith("params.npz") and "average_l0_" in root:
                    # root contains average_l0_XX
                    m = re.search(r"average_l0_(\d+)", root)
                    val = int(m.group(1)) if m else 0
                    cand.append((val, os.path.join(root, f)))
        cand.sort()
        if cand:
            idx = max(0, min(len(cand)-1, int(len(cand)*SELECTION_PERCENTILE)))
            return cand[idx][1]

    # Then try online (optional)
    if not ALLOW_ONLINE_DOWNLOAD or hf_hub_download is None or HfApi is None:
        return None

    api = HfApi()
    pref = f"layer_{layer_idx}/{WIDTH_DIR}/average_l0_"
    files = api.list_repo_files(repo_id=BASE_REPO_ID)
    cands = [(int(re.search(r"average_l0_(\d+)/", f).group(1)), f)
             for f in files if f.startswith(pref) and f.endswith("params.npz")]
    cands.sort()
    if not cands:
        return None
    idx = max(0, min(len(cands)-1, int(len(cands)*SELECTION_PERCENTILE)))
    filename = cands[idx][1]
    local_path = hf_hub_download(repo_id=BASE_REPO_ID, filename=filename)
    return local_path

def fuse_one_layer(layer_dir: str, out_dir: str):
    # delta pt path
    # Directory like layer_20_dnew_32_pctl_0.50/, containing best_model.pt
    pt_path = os.path.join(layer_dir, "best_model.pt")
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"delta pt not found: {pt_path}")

    # Layer index parsing
    m = re.search(r"layer_(\d+)_", os.path.basename(layer_dir))
    if not m:
        raise RuntimeError(f"cannot parse layer index from {layer_dir}")
    layer_idx = int(m.group(1))

    # base npz
    base_npz = load_base_npz_path(layer_idx)
    if not base_npz:
        raise FileNotFoundError(f"base npz not available for layer {layer_idx} (prepare LOCAL_BASE_ROOT or enable ALLOW_ONLINE_DOWNLOAD)")

    # Load base
    base = np.load(base_npz)
    W_enc_b = base["W_enc"]  # [d_model, d_base]
    b_enc_b = base["b_enc"]  # [d_base]
    W_dec_b = base["W_dec"]  # [d_base, d_model]
    b_dec_b = base["b_dec"]  # [d_model]
    thr_b   = base["threshold"]  # [d_base]
    d_model, d_base = W_enc_b.shape[0], W_enc_b.shape[1]

    # Load delta
    sd = torch.load(pt_path, map_location="cpu")
    # Your DeltaTranscoder.state_dict naming: base and delta_features two sets of parameters
    # We only take the weights of delta_features and treat it as a new JumpReLU
    We_d = sd["delta_features.W_enc"].numpy()      # [d_model, d_new]
    be_d = sd["delta_features.b_enc"].numpy()      # [d_new]
    Wd_d = sd["delta_features.W_dec"].numpy()      # [d_new, d_model]
    bd_d = sd["delta_features.b_dec"].numpy()      # [d_model]
    th_d = sd["delta_features.threshold"].numpy()  # [d_new]
    d_new = We_d.shape[1]

    assert We_d.shape[0] == d_model
    assert Wd_d.shape[1] == d_model

    # Fuse
    W_enc = np.concatenate([W_enc_b, We_d], axis=1)                # [d_model, d_base+d_new]
    b_enc = np.concatenate([b_enc_b, be_d], axis=0)                # [d_base+d_new]
    W_dec = np.concatenate([W_dec_b, Wd_d], axis=0)                # [d_base+d_new, d_model]
    b_dec = (b_dec_b + bd_d).astype(W_dec.dtype)                   # [d_model]
    thr   = np.concatenate([thr_b, th_d], axis=0)                  # [d_base+d_new]

    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "params.npz"),
             W_enc=W_enc, b_enc=b_enc, W_dec=W_dec, b_dec=b_dec, threshold=thr)

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    # Iterate over each layer in the delta directory
    layer_dirs = [os.path.join(DELTA_ROOT, d) for d in os.listdir(DELTA_ROOT)
                  if d.startswith("layer_") and os.path.isdir(os.path.join(DELTA_ROOT, d))]
    layer_dirs.sort(key=lambda p: int(re.search(r"layer_(\d+)_", os.path.basename(p)).group(1)))

    for ld in layer_dirs:
        li = int(re.search(r"layer_(\d+)_", os.path.basename(ld)).group(1))
        out_dir = os.path.join(OUT_ROOT, f"layer_{li}")
        fuse_one_layer(ld, out_dir)
        print(f"[OK] fused layer {li} -> {out_dir}/params.npz")

    # Write local config.yaml (no online needed)
    cfg_path = os.path.join(OUT_ROOT, "config.yaml")
    with open(cfg_path, "w") as f:
        f.write("model_name: \"google/gemma-2-2b\"\n")
        f.write("model_kind: \"transcoder_set\"\n")
        f.write("feature_input_hook: \"hook_mlp_in\"\n")
        f.write("feature_output_hook: \"hook_mlp_out\"\n")
        f.write("transcoders:\n")
        for ld in layer_dirs:
            li = int(re.search(r"layer_(\d+)_", os.path.basename(ld)).group(1))
            f.write(f"  - \"{os.path.join(OUT_ROOT, f'layer_{li}', 'params.npz')}\"\n")

    print(f"[DONE] wrote {cfg_path}")

if __name__ == "__main__":
    main()