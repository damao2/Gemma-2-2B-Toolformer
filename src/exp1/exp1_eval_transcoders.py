import os
import math
import csv
import json
import glob
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download, HfApi
import transformer_lens
import matplotlib.pyplot as plt

# ================================================================
# 1. SAE / DeltaTranscoder 模块定义
# ================================================================

class JumpReLU_Module(nn.Module):
    """Gemma-Scope Transcoder / SAE 基本模块（与训练脚本一致）"""
    def __init__(self, d_model, d_sae, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.d_model, self.d_sae = d_model, d_sae
        self.W_enc = nn.Parameter(torch.empty((d_model, d_sae), **factory_kwargs))
        self.b_enc = nn.Parameter(torch.empty(d_sae, **factory_kwargs))
        self.W_dec = nn.Parameter(torch.empty((d_sae, d_model), **factory_kwargs))
        self.b_dec = nn.Parameter(torch.empty(d_model, **factory_kwargs))
        # threshold 只读
        self.threshold = nn.Parameter(torch.empty(d_sae, **factory_kwargs), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.W_enc.dtype)
        pre_acts = x @ self.W_enc + self.b_enc
        acts = F.relu(pre_acts) * (pre_acts > self.threshold).float()
        return acts @ self.W_dec + self.b_dec


class DeltaTranscoder(nn.Module):
    """封装 base SAE + delta SAE（用于推理时输出 base+delta 重建）"""
    def __init__(self, base_transcoder: JumpReLU_Module, d_new: int):
        super().__init__()
        self.base_transcoder = base_transcoder
        self.d_model = base_transcoder.d_model
        self.delta_features = JumpReLU_Module(
            self.d_model, d_new,
            device=next(base_transcoder.parameters()).device,
            dtype=next(base_transcoder.parameters()).dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(next(self.base_transcoder.parameters()).dtype)
        with torch.no_grad():
            base_recon = self.base_transcoder(x)
            delta_recon = self.delta_features(x)
        return base_recon + delta_recon


# ================================================================
# 2. 加载器与辅助函数
# ================================================================

def select_base_filename_from_hub(
    repo_id: str, layer_idx: int, width_dir: str, percentile: float
) -> str:
    """从 HF Hub 选出给定层、给定 percentile 的 base SAE .npz 文件名"""
    print(f"[HF] Search base SAE in {repo_id}, layer={layer_idx}, width={width_dir} ...")
    api = HfApi()
    pref = f"layer_{layer_idx}/{width_dir}/average_l0_"
    files = api.list_repo_files(repo_id=repo_id)
    cands: List[Tuple[int, str]] = []
    for f in files:
        if f.startswith(pref) and f.endswith("params.npz"):
            k = int(f.split("average_l0_")[1].split("/")[0])
            cands.append((k, f))
    if not cands:
        raise FileNotFoundError(f"No base transcoders found for layer {layer_idx} in {repo_id}")
    cands.sort()
    index = max(0, min(len(cands) - 1, int(len(cands) * percentile)))
    filename = cands[index][1]
    print(f"[HF] Selected base SAE ({percentile:.0%}): {filename}")
    return filename


def load_base_transcoder_from_hub(
    repo_id: str, filename: str, device: torch.device
) -> JumpReLU_Module:
    """加载原始 Gemma-Scope SAE 参数到 JumpReLU_Module"""
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    params = np.load(path)
    d_model = params["b_dec"].shape[0]
    d_sae = params["b_enc"].shape[0]
    sae = JumpReLU_Module(d_model, d_sae, device=device, dtype=torch.float32)
    state_dict = {k: torch.from_numpy(v) for k, v in params.items()}
    sae.load_state_dict(state_dict)
    sae.eval()
    sae.requires_grad_(False)
    return sae


def find_delta_checkpoint(
    delta_root: str, layer: int, d_new: int, percentile: float
) -> str:
    """
    根据 layer, d_new, percentile 在 delta_root 下自动查找 best_model.pt。
    默认匹配模式: layer_{L}_dnew_{d_new}_pctl_* / best_model.pt
    你可以根据实际目录结构修改此函数。
    """
    p_str = f"{percentile:.2f}"
    # p_str=percentile
    # 尝试更具体的前缀
    pattern1 = os.path.join(
        delta_root, f"dnew_{d_new}_pctl_{p_str}", f"layer_{layer}_dnew_{d_new}_pctl_{p_str}", "best_model.pt"
    )
    if os.path.exists(pattern1):
        return pattern1
    # 回退: 用通配符匹配
    pattern2 = os.path.join(delta_root, f"dnew_{d_new}_pctl_*", f"layer_{layer}_dnew_{d_new}_pctl_*", "best_model.pt")
    matches = glob.glob(pattern2)
    if not matches:
        raise FileNotFoundError(f"No delta checkpoint found for layer {layer}, d_new={d_new}, "
                                f"p={percentile} under {delta_root}")
    if len(matches) > 1:
        print(f"[WARN] Multiple delta checkpoints found, using first:\n  " +
              "\n  ".join(matches))
    return matches[0]


def load_finetuned_delta_transcoder(
    finetuned_path: str, base_transcoder: JumpReLU_Module,
    d_new: int, device: torch.device
) -> DeltaTranscoder:
    """加载你训练好的 DeltaTranscoder state_dict（best_model.pt）"""
    print(f"[Delta] Loading finetuned DeltaTranscoder: {finetuned_path}")
    model = DeltaTranscoder(base_transcoder, d_new).to(device)
    state_dict = torch.load(finetuned_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.requires_grad_(False)
    return model


def tokenize_prompts(tokenizer, eval_file: str, device: torch.device) -> List[torch.Tensor]:
    """读取 eval_file，每行一个 prompt，标准化换行并编码为 input_ids 张量"""
    prompts: List[str] = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            # 把字面 "\n" 变为真实换行，并确保末尾有一个换行
            text = line.replace("\\n", "\n").rstrip() + "\n"
            prompts.append(text)

    token_batches: List[torch.Tensor] = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt")["input_ids"].to(device)
        token_batches.append(ids)
    print(f"[Eval] Loaded {len(token_batches)} prompts from {eval_file}")
    return token_batches


def ce_on_dataset(
    model: transformer_lens.HookedTransformer,
    token_batches: List[torch.Tensor],
    fwd_hooks: List[Tuple[str, callable]] = None,
) -> float:
    """
    在整个数据集上计算 token-level 平均 CE。
    如果提供 fwd_hooks，则在该 hook 环境下前向。
    """
    total_loss = 0.0
    total_tokens = 0
    for ids in token_batches:
        with torch.no_grad():
            if fwd_hooks is None:
                logits = model(ids, return_type="logits")
            else:
                with model.hooks(fwd_hooks=fwd_hooks):
                    logits = model(ids, return_type="logits")
            # shift CE（忽略最后一个 token 的预测）
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                ids[:, 1:].reshape(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += ids[:, 1:].numel()
    return total_loss / total_tokens


# ================================================================
# 3. Hook 创建与层重建 metrics 累积
# ================================================================

def create_hooks_for_single_layer(
    layer_idx: int,
    transcoder_module: nn.Module,
    metrics: Dict[int, Dict],
    use_norm: bool = True,
) -> List[Tuple[str, callable]]:
    """
    为单层创建 hook:
      - 捕获 hook_mlp_in 输入
      - 替换 hook_mlp_out 输出为 SAE 重建
      - 同时累积 cos / l2_ratio / mean / std 等指标到 metrics[layer_idx]
    """
    fwd_hooks: List[Tuple[str, callable]] = []
    context: Dict[str, torch.Tensor] = {}
    d_model = transcoder_module.d_model

    def capture_input_hook(mlp_input: torch.Tensor, hook) -> torch.Tensor:
        context["x"] = mlp_input
        return mlp_input

    def replace_output_hook(mlp_output: torch.Tensor, hook, tm=transcoder_module) -> torch.Tensor:
        x = context.get("x", None)
        if x is None:
            raise ValueError(f"[Hook] MLP input not captured for layer {layer_idx}")

        if use_norm:
            norm_factor = math.sqrt(d_model)
            x_norm = x / norm_factor
            recon = tm(x_norm)
            recon = recon * norm_factor
        else:
            recon = tm(x)

        # 统计重建质量
        with torch.no_grad():
            B, T, D = mlp_output.shape
            orig = mlp_output.reshape(-1, D)
            rec = recon.reshape(-1, D)
            cos = F.cosine_similarity(orig, rec, dim=-1)  # [B*T]
            diff = rec - orig
            l2_num = diff.pow(2).sum(-1).sqrt()
            l2_den = orig.pow(2).sum(-1).sqrt().clamp_min(1e-8)
            l2_ratio = l2_num / l2_den

            m = metrics.setdefault(
                layer_idx,
                dict(
                    cos_sum=0.0,
                    l2_sum=0.0,
                    orig_mean_sum=0.0,
                    orig_std_sum=0.0,
                    rec_mean_sum=0.0,
                    rec_std_sum=0.0,
                    n_tokens=0,
                    n_batches=0,
                ),
            )
            m["cos_sum"] += float(cos.sum().item())
            m["l2_sum"] += float(l2_ratio.sum().item())
            m["n_tokens"] += cos.numel()
            m["orig_mean_sum"] += float(mlp_output.mean().item())
            m["orig_std_sum"] += float(mlp_output.std().item())
            m["rec_mean_sum"] += float(recon.mean().item())
            m["rec_std_sum"] += float(recon.std().item())
            m["n_batches"] += 1

        # 保留 BOS 位置不变
        if x.shape[1] > 0:
            recon[:, 0, :] = mlp_output[:, 0, :]
        return recon.to(mlp_output.dtype)

    fwd_hooks.append((f"blocks.{layer_idx}.hook_mlp_in", capture_input_hook))
    fwd_hooks.append((f"blocks.{layer_idx}.hook_mlp_out", replace_output_hook))
    return fwd_hooks


def summarize_layer_metrics(m: Dict) -> Dict:
    """将累积量转成平均指标"""
    n_tok = max(1, m.get("n_tokens", 0))
    n_batch = max(1, m.get("n_batches", 0))
    return dict(
        mean_cos=m["cos_sum"] / n_tok,
        mean_l2_ratio=m["l2_sum"] / n_tok,
        orig_mean=m["orig_mean_sum"] / n_batch,
        orig_std=m["orig_std_sum"] / n_batch,
        rec_mean=m["rec_mean_sum"] / n_batch,
        rec_std=m["rec_std_sum"] / n_batch,
        n_tokens=m["n_tokens"],
        n_batches=m["n_batches"],
    )


# ================================================================
# 4. 主实验流程
# ================================================================

def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------- 4.1 加载微调后的 Gemma 模型 ----------
    print("[Model] Loading tokenizer & finetuned base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model = transformer_lens.HookedTransformer.from_pretrained(
        "google/gemma-2-2b",
        torch_dtype=torch.bfloat16,
        fold_ln=True,
        center_writing_weights=False,
        center_unembed=False,
        device=device,
        hf_model=hf_base,
        tokenizer=tokenizer,
    )
    model.set_use_hook_mlp_in(True)
    # model.set_use_hook_mlp_out(True)
    model.eval()

    print("[Eval] Tokenizing prompts...")
    token_batches = tokenize_prompts(tokenizer, args.eval_file, device)

    # ---------- 4.2 Raw (不挂 SAE) ----------
    print("[Eval] Computing RAW CE over dataset...")
    ce_raw = ce_on_dataset(model, token_batches, fwd_hooks=None)
    print(f"[RAW] CE = {ce_raw:.6f}")

    # 层列表
    if args.layers == "all":
        layer_list = list(range(model.cfg.n_layers))
    else:
        layer_list = [int(x) for x in args.layers.split(",") if x.strip()]

    print(f"[Layers] Will evaluate layers: {layer_list}")

    # 存放所有结果
    rows: List[Dict] = []

    # ---------- 4.3 对每一层做 Base-only & Full (base+delta) ----------
    for L in layer_list:
        print(f"\n========== Layer {L} ==========")

        # 4.3.1 Base-only: 原始 Gemma-Scope SAE
        print(f"[Base] Loading base SAE for layer {L} ...")
        base_filename = select_base_filename_from_hub(
            args.sae_repo, L, args.sae_width_dir, args.percentile
        )
        base_sae = load_base_transcoder_from_hub(args.sae_repo, base_filename, device)

        # 该层的 metrics 累积
        base_metrics: Dict[int, Dict] = {}

        base_hooks = create_hooks_for_single_layer(
            layer_idx=L,
            transcoder_module=base_sae,
            metrics=base_metrics,
            use_norm=True,
        )

        print(f"[Base] Evaluating CE with base-only SAE on layer {L} ...")
        ce_base = ce_on_dataset(model, token_batches, fwd_hooks=base_hooks)
        print(f"[Base] Layer {L}: CE = {ce_base:.6f}, ΔCE = {ce_base - ce_raw:+.6f}")

        base_layer_summary = summarize_layer_metrics(base_metrics[L])

        # 4.3.2 Full: base + delta（DeltaTranscoder）
        print(f"[Full] Loading finetuned DeltaTranscoder for layer {L} ...")
        delta_ckpt = find_delta_checkpoint(args.delta_root, L, args.d_new, args.percentile)
        delta_model = load_finetuned_delta_transcoder(
            finetuned_path=delta_ckpt,
            base_transcoder=base_sae,
            d_new=args.d_new,
            device=device,
        )

        full_metrics: Dict[int, Dict] = {}
        full_hooks = create_hooks_for_single_layer(
            layer_idx=L,
            transcoder_module=delta_model,
            metrics=full_metrics,
            use_norm=True,
        )

        print(f"[Full] Evaluating CE with base+delta SAE on layer {L} ...")
        ce_full = ce_on_dataset(model, token_batches, fwd_hooks=full_hooks)
        print(f"[Full] Layer {L}: CE = {ce_full:.6f}, ΔCE = {ce_full - ce_raw:+.6f}")

        full_layer_summary = summarize_layer_metrics(full_metrics[L])

        # 汇总一行
        row = dict(
            layer=L,
            d_new=args.d_new,
            percentile=args.percentile,
            ce_raw=ce_raw,
            ce_base=ce_base,
            ce_full=ce_full,
            dce_base=ce_base - ce_raw,
            dce_full=ce_full - ce_raw,
            base_mean_cos=base_layer_summary["mean_cos"],
            base_mean_l2=base_layer_summary["mean_l2_ratio"],
            base_orig_mean=base_layer_summary["orig_mean"],
            base_orig_std=base_layer_summary["orig_std"],
            base_rec_mean=base_layer_summary["rec_mean"],
            base_rec_std=base_layer_summary["rec_std"],
            full_mean_cos=full_layer_summary["mean_cos"],
            full_mean_l2=full_layer_summary["mean_l2_ratio"],
            full_orig_mean=full_layer_summary["orig_mean"],
            full_orig_std=full_layer_summary["orig_std"],
            full_rec_mean=full_layer_summary["rec_mean"],
            full_rec_std=full_layer_summary["rec_std"],
            n_tokens_base=base_layer_summary["n_tokens"],
            n_tokens_full=full_layer_summary["n_tokens"],
        )
        rows.append(row)

    # ---------- 4.4 保存 CSV / JSON ----------
    tag = f"d{args.d_new}_p{args.percentile}"
    csv_path = os.path.join(args.output_dir, f"exp1_metrics_{tag}.csv")
    json_path = os.path.join(args.output_dir, f"exp1_metrics_{tag}.json")

    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"[Save] Metrics saved to:\n  {csv_path}\n  {json_path}")
    else:
        print("[WARN] No rows collected — check layers / eval_file / paths.")
        return

    # ---------- 4.5 画图并保存 ----------
    layers = [r["layer"] for r in rows]
    dce_base = [r["dce_base"] for r in rows]
    dce_full = [r["dce_full"] for r in rows]
    base_cos = [r["base_mean_cos"] for r in rows]
    full_cos = [r["full_mean_cos"] for r in rows]

    # CE vs layer
    plt.figure(figsize=(8, 4))
    plt.plot(layers, dce_base, "o-", label="ΔCE base-only")
    plt.plot(layers, dce_full, "o-", label="ΔCE base+delta")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Layer")
    plt.ylabel("ΔCE (vs RAW)")
    plt.title(f"Exp1 ΔCE per layer (d_new={args.d_new}, p={args.percentile})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    ce_fig_path = os.path.join(args.output_dir, f"exp1_dce_vs_layer_{tag}.png")
    plt.tight_layout()
    plt.savefig(ce_fig_path)
    plt.close()
    print(f"[Save] ΔCE figure saved to: {ce_fig_path}")

    # cos vs layer
    plt.figure(figsize=(8, 4))
    plt.plot(layers, base_cos, "o-", label="cos base-only")
    plt.plot(layers, full_cos, "o-", label="cos base+delta")
    plt.xlabel("Layer")
    plt.ylabel("mean cos(orig, recon)")
    plt.title(f"Exp1 cosine per layer (d_new={args.d_new}, p={args.percentile})")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    cos_fig_path = os.path.join(args.output_dir, f"exp1_cos_vs_layer_{tag}.png")
    plt.tight_layout()
    plt.savefig(cos_fig_path)
    plt.close()
    print(f"[Save] Cosine figure saved to: {cos_fig_path}")


# ================================================================
# 5. CLI
# ================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Experiment 1: per-layer base vs delta SAE evaluation")
    p.add_argument("--model_name", type=str,
                   default="./gemma_2b_toolformer_merged_v4",
                   help="Path or HF name of finetuned Gemma model")
    p.add_argument("--eval_file", type=str, required=True,
                   help="Path to eval prompts txt (one prompt per line)")
    p.add_argument("--output_dir", type=str, default="./exp1_outputs",
                   help="Where to save metrics & plots")

    # 原始 Gemma-Scope SAE repo
    p.add_argument("--sae_repo", type=str,
                   default="google/gemma-scope-2b-pt-transcoders")
    p.add_argument("--sae_width_dir", type=str, default="width_16k")
    p.add_argument("--percentile", type=float, required=True,
                   help="Percentile used to select base SAE (e.g. 0.50)")

    # DeltaTranscoder 配置
    p.add_argument("--delta_root", type=str, required=True,
                   help="Root dir containing finetuned DeltaTranscoder checkpoints")
    p.add_argument("--d_new", type=int, required=True,
                   help="Number of new features (delta dim)")

    # 层选择: "all" 或 "0,1,2,..." 
    p.add_argument("--layers", type=str, default="all",
                   help='Layers to eval: "all" or comma list like "16,20,22"')

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)

'''
python3 exp1_eval_transcoders.py \
  --model_name /home/users/qc62/MI/gemma_2b_toolformer_merged_v4 \
  --eval_file /home/users/qc62/MI/gen_dataset_SAE/exp1_sae_validation.txt \
  --delta_root /usr/project/xtmp/qc62/train_4 \
  --d_new 128 \
  --percentile 0.75 \
  --layers all \
  --output_dir ./exp1/exp1_outputs_d128_p0.75
'''