
# filepath: /home/users/qc62/exp2/exp2A_eval_no_delta.py
import os
import re
import math
import csv
import json
import glob
import argparse
from typing import Dict, List, Tuple, Any, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download, HfApi
import transformer_lens
import matplotlib.pyplot as plt

# ================================================================
# 1. SAE / DeltaTranscoder 基本模块
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
    """
    base SAE + delta SAE（fused SAE）
    注意：在本实验里我们不会直接调用它的 forward，而是在 hook 里手动算 base/delta。
    """
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
# 2. SAE / Delta 加载 & 数据加载
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
    目录假定大致形如:
      /.../dnew_{d_new}_pctl_0.50/layer_{L}_dnew_{d_new}_pctl_0.50/best_model.pt
    """
    p_str = f"{percentile:.2f}"
    pattern1 = os.path.join(
        delta_root,
        f"dnew_{d_new}_pctl_{p_str}",
        f"layer_{layer}_dnew_{d_new}_pctl_{p_str}",
        "best_model.pt",
    )
    if os.path.exists(pattern1):
        return pattern1
    pattern2 = os.path.join(
        delta_root,
        f"dnew_{d_new}_pctl_*",
        f"layer_{layer}_dnew_{d_new}_pctl_*",
        "best_model.pt",
    )
    matches = glob.glob(pattern2)
    if not matches:
        raise FileNotFoundError(
            f"No delta checkpoint found for layer {layer}, d_new={d_new}, "
            f"p={percentile} under {delta_root}"
        )
    if len(matches) > 1:
        print("[WARN] Multiple delta checkpoints found, using first:\n  " +
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


def load_jsonl_by_label(
    path: str,
    tool_labels: Set[str],
    copy_labels: Set[str],
    tool_ood_labels: Set[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    读取 JSONL，每行至少包含: input, target, label
    根据 label 分成子集: tool, copy, tool_ood
    """
    subsets: Dict[str, List[Dict[str, Any]]] = {
        "tool": [],
        "copy": [],
        "tool_ood": [],
    }
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            label = str(ex.get("label", "")).lower()
            if label in tool_labels:
                subsets["tool"].append(ex)
            elif label in copy_labels:
                subsets["copy"].append(ex)
            elif label in tool_ood_labels:
                subsets["tool_ood"].append(ex)
    for name, data in subsets.items():
        print(f"[Data] subset={name}, num_examples={len(data)}")
    return subsets


# ================================================================
# 3. Hook 创建：full / no_delta + 激活统计
# ================================================================

def create_hooks_for_layer_mode(
    layer_idx: int,
    fused_sae: DeltaTranscoder,
    metrics: Dict[str, float],
    mode: str,
    use_norm: bool = True,
) -> List[Tuple[str, callable]]:
    """
    为单层创建 hook，用 fused SAE 的 base+delta：
      - mode='full'    : 重建 = base + delta
      - mode='no_delta': 重建 = base （但仍统计 delta 激活）
    同时统计:
      - mean_abs_base_acts, mean_abs_delta_acts
    累积到 metrics dict 中（在 eval 时按 subset 调用一次）。
    """
    assert mode in ("full", "no_delta")
    fwd_hooks: List[Tuple[str, callable]] = []
    context: Dict[str, torch.Tensor] = {}
    d_model = fused_sae.d_model
    base = fused_sae.base_transcoder
    delta = fused_sae.delta_features

    def capture_input_hook(mlp_input: torch.Tensor, hook) -> torch.Tensor:
        context["x"] = mlp_input
        return mlp_input

    def replace_output_hook(mlp_output: torch.Tensor, hook) -> torch.Tensor:
        x = context.get("x", None)
        if x is None:
            raise ValueError(f"[Hook] MLP input not captured for layer {layer_idx}")

        if use_norm:
            norm_factor = math.sqrt(d_model)
            x_norm = x / norm_factor
        else:
            norm_factor = 1.0
            x_norm = x

        # 手动计算 base / delta acts & recon
        x_norm = x_norm.to(base.W_enc.dtype)
        pre_b = x_norm @ base.W_enc + base.b_enc
        acts_b = F.relu(pre_b) * (pre_b > base.threshold).float()
        base_recon_norm = acts_b @ base.W_dec + base.b_dec

        pre_d = x_norm @ delta.W_enc + delta.b_enc
        acts_d = F.relu(pre_d) * (pre_d > delta.threshold).float()
        delta_recon_norm = acts_d @ delta.W_dec + delta.b_dec

        if mode == "full":
            recon_norm = base_recon_norm + delta_recon_norm
        else:  # no_delta
            recon_norm = base_recon_norm

        recon = recon_norm * norm_factor

        # 统计激活规模
        with torch.no_grad():
            metrics.setdefault("base_abs_act_sum", 0.0)
            metrics.setdefault("delta_abs_act_sum", 0.0)
            metrics.setdefault("n_tokens", 0)

            B, T, _ = acts_b.shape
            metrics["base_abs_act_sum"] += float(acts_b.abs().sum().item())
            metrics["delta_abs_act_sum"] += float(acts_d.abs().sum().item())
            metrics["n_tokens"] += B * T

        # BOS 位置保持不变
        if x.shape[1] > 0:
            recon[:, 0, :] = mlp_output[:, 0, :]
        return recon.to(mlp_output.dtype)

    fwd_hooks.append((f"blocks.{layer_idx}.hook_mlp_in", capture_input_hook))
    fwd_hooks.append((f"blocks.{layer_idx}.hook_mlp_out", replace_output_hook))
    return fwd_hooks


# ================================================================
# 4. 工具调用解析 & 评估：CE + 工具调用率 + 数学正确率
# ================================================================

TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*calculator\(([^)]+)\)\s*</tool_call>", re.DOTALL
)

def extract_expr_from_tool_call(text: str) -> Optional[str]:
    """
    从文本中提取 <tool_call>calculator(expr)</tool_call> 里的 expr。
    若没有匹配则返回 None。
    """
    m = TOOL_CALL_PATTERN.search(text)
    if not m:
        return None
    expr = m.group(1).strip()
    return expr or None


def safe_eval_expr(expr: str) -> Optional[float]:
    """
    安全地对形如 '123+45*6' 的表达式求值，只允许 0-9 + - * / 与空白。
    若非法或 eval 失败，返回 None。
    """
    if not re.fullmatch(r"[0-9+\-*/\s]+", expr):
        return None
    try:
        val = eval(expr, {"__builtins__": {}}, {})
    except Exception:
        return None
    if not isinstance(val, (int, float)):
        return None
    return float(val)

def generate_with_hooks(
    model: transformer_lens.HookedTransformer,
    tokenizer: AutoTokenizer,
    prompt: str,
    fwd_hooks: Optional[List[Tuple[str, callable]]],
    device: torch.device,
    max_new_tokens: int = 64,
) -> str:
    """给定 prompt，在 (可选) hooks 下做 greedy 生成，返回新生成部分的文本。"""
    with torch.no_grad():
        tokens = model.to_tokens(prompt, prepend_bos=True).to(device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            # do_sample=False,          # greedy
            # pad_token_id=tokenizer.eos_token_id,
            temperature=0.0
        )

        if fwd_hooks is None:
            out_ids = model.generate(tokens, **gen_kwargs)
        else:
            with model.hooks(fwd_hooks=fwd_hooks):
                out_ids = model.generate(tokens, **gen_kwargs)

    # 只取新生成的部分
    gen_ids = out_ids[0, tokens.size(1):]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    return gen_text

def eval_subset_for_condition(
    model: transformer_lens.HookedTransformer,
    tokenizer: AutoTokenizer,
    examples: List[Dict[str, Any]],
    fwd_hooks: Optional[List[Tuple[str, callable]]],
    device: torch.device,
    condition: str,
    gen_log_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    - mean_ce：teacher forcing
    - 其它三个 metric 基于真实生成
    - 若 gen_log_path 不为 None，则把每条样本的生成写到该 JSONL 文件
    """
    total_loss = 0.0
    total_tokens = 0
    n_examples = 0

    tool_count = 0
    valid_tool_count = 0
    math_correct = 0
    math_total = 0  # 有 gold 表达式的样本数

    for ex in examples:
        prompt = str(ex["input"])
        target = str(ex["target"])

        # 编码：分别编码 prompt 与 target，然后拼接
        with torch.no_grad():
            prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
            target_ids = tokenizer(target, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

        # [1, P] , [1, T]
        P = prompt_ids.size(1)
        T = target_ids.size(1)
        input_ids = torch.cat([prompt_ids, target_ids], dim=1)  # [1, P+T]

        with torch.no_grad():
            if fwd_hooks is None:
                logits = model(input_ids, return_type="logits")  # [1, P+T, V]
            else:
                with model.hooks(fwd_hooks=fwd_hooks):
                    logits = model(input_ids, return_type="logits")

        # 只在 target 区间上计算 CE:
        logits_target = logits[:, P-1 : P+T-1, :]          # [1, T, V]
        labels_target = input_ids[:, P : P+T]              # [1, T]

        loss = F.cross_entropy(
            logits_target.reshape(-1, logits_target.size(-1)),
            labels_target.reshape(-1),
            reduction="sum",
        )
        total_loss += float(loss.item())
        total_tokens += int(labels_target.numel())
        n_examples += 1

        prompt_for_gen = prompt if prompt.endswith("\n") else prompt + "\n"
        gen_text = generate_with_hooks(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_for_gen,
            fwd_hooks=fwd_hooks,
            device=device,
            max_new_tokens=64,  # 可以按你 target 长度调
        )
        
        if gen_log_path is not None:
            with open(gen_log_path, "a", encoding="utf-8") as gf:
                gf.write(json.dumps({
                    "condition": condition,
                    "input": prompt,
                    "target": target,
                    "generation": gen_text,
                }, ensure_ascii=False) + "\n")


        if "<tool_call>" in gen_text:
            tool_count += 1

        if re.search(r"<tool_call>.*?</tool_call>", gen_text, flags=re.DOTALL):
            valid_tool_count += 1

        # ---------- 数学正确率 ----------
        # 从 gold target 与 pred_text 中提取 expr，并比较数值是否一致
        gold_expr = extract_expr_from_tool_call(target)
        if gold_expr is not None:
            gold_val = safe_eval_expr(gold_expr)
            if gold_val is not None:
                pred_expr = extract_expr_from_tool_call(gen_text)
                pred_val = safe_eval_expr(pred_expr) if pred_expr is not None else None
                if pred_val is not None and abs(pred_val - gold_val) < 1e-6:
                    math_correct += 1
                math_total += 1

    if total_tokens == 0 or n_examples == 0:
        return dict(
            mean_ce=float("nan"),
            tool_call_rate=float("nan"),
            valid_tool_format_rate=float("nan"),
            math_correct_rate=float("nan"),
            n_examples=n_examples,
            n_tokens=total_tokens,
        )

    mean_ce = total_loss / total_tokens
    tool_rate = tool_count / n_examples if n_examples > 0 else float("nan")
    valid_rate = valid_tool_count / n_examples if n_examples > 0 else float("nan")
    math_rate = (math_correct / math_total) if math_total > 0 else float("nan")

    return dict(
        mean_ce=mean_ce,
        tool_call_rate=tool_rate,
        valid_tool_format_rate=valid_rate,
        math_correct_rate=math_rate,
        n_examples=n_examples,
        n_tokens=total_tokens,
    )


# ================================================================
# 5. 主实验：per-layer, per-subset, per-condition
# ================================================================

def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------- 5.1 加载模型 ----------
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

    gen_dir = os.path.join(args.output_dir, "generations")
    os.makedirs(gen_dir, exist_ok=True)
    # ---------- 5.2 加载验证集 & 按 label 分子集 ----------
    tool_labels = set([s.strip().lower() for s in args.tool_labels.split(",") if s.strip()])
    copy_labels = set([s.strip().lower() for s in args.copy_labels.split(",") if s.strip()])
    tool_ood_labels = set([s.strip().lower() for s in args.tool_ood_labels.split(",") if s.strip()])
    print(f"[Labels] tool={tool_labels}, copy={copy_labels}, tool_ood={tool_ood_labels}")

    subsets = load_jsonl_by_label(
        args.val_file,
        tool_labels=tool_labels,
        copy_labels=copy_labels,
        tool_ood_labels=tool_ood_labels,
    )

    # 层列表
    if args.layers == "all":
        layer_list = list(range(model.cfg.n_layers))
    else:
        layer_list = [int(x) for x in args.layers.split(",") if x.strip()]

    print(f"[Layers] Will evaluate layers: {layer_list}")

    # ---------- 5.3 结果累积 ----------
    rows: List[Dict[str, Any]] = []

    # ---------- 5.4 Raw baseline（和层无关，只按 subset 算一遍） ----------
    raw_stats_by_subset: Dict[str, Dict[str, float]] = {}
    for subset_name, exs in subsets.items():
        if not exs:
            continue
        print(f"[RAW] Evaluating subset={subset_name} ...")
        raw_gen_log = os.path.join(gen_dir, f"layer_raw_{subset_name}_raw.jsonl")
        if os.path.exists(raw_gen_log):
            os.remove(raw_gen_log)
        stats = eval_subset_for_condition(
            model=model,
            tokenizer=tokenizer,
            examples=exs,
            fwd_hooks=None,
            device=device,
            condition="raw",
            gen_log_path=raw_gen_log,
        )
        raw_stats_by_subset[subset_name] = stats
        print(f"[RAW] subset={subset_name}, mean_ce={stats['mean_ce']:.6f}, "
              f"tool_call_rate={stats['tool_call_rate']:.3f}, "
              f"valid_tool_format_rate={stats['valid_tool_format_rate']:.3f}, "
              f"math_correct_rate={stats['math_correct_rate']:.3f}")

    # ---------- 5.5 对每一层: full / no_delta ----------
    for L in layer_list:
        print(f"\n========== Layer {L} ==========")

        # 5.5.1 加载 base SAE & fused SAE
        print(f"[Base] Loading base SAE for layer {L} ...")
        base_filename = select_base_filename_from_hub(
            args.sae_repo, L, args.sae_width_dir, args.percentile
        )
        base_sae = load_base_transcoder_from_hub(args.sae_repo, base_filename, device)

        print(f"[Full] Loading finetuned DeltaTranscoder for layer {L} ...")
        delta_ckpt = find_delta_checkpoint(args.delta_root, L, args.d_new, args.percentile)
        fused_sae = load_finetuned_delta_transcoder(
            finetuned_path=delta_ckpt,
            base_transcoder=base_sae,
            d_new=args.d_new,
            device=device,
        )

        # 每个 subset, 每种 condition: raw / full / no_delta
        for subset_name, exs in subsets.items():
            if not exs:
                continue

            # RAW 已经算过
            raw_stats = raw_stats_by_subset.get(subset_name, None)

            # full（base+delta）
            full_act_metrics: Dict[str, float] = {}
            full_hooks = create_hooks_for_layer_mode(
                layer_idx=L,
                fused_sae=fused_sae,
                metrics=full_act_metrics,
                mode="full",
                use_norm=True,
            )
            print(f"[Full] Evaluating subset={subset_name}, layer={L} ...")
            full_gen_log = os.path.join(gen_dir, f"layer_{L}_{subset_name}_full.jsonl")
            if os.path.exists(full_gen_log):
                os.remove(full_gen_log)
            full_stats = eval_subset_for_condition(
                model=model,
                tokenizer=tokenizer,
                examples=exs,
                fwd_hooks=full_hooks,
                device=device,
                condition="full",
                gen_log_path=full_gen_log,
            )

            # no_delta（仅 base 重建）
            no_delta_act_metrics: Dict[str, float] = {}
            no_delta_hooks = create_hooks_for_layer_mode(
                layer_idx=L,
                fused_sae=fused_sae,
                metrics=no_delta_act_metrics,
                mode="no_delta",
                use_norm=True,
            )
            print(f"[NoDelta] Evaluating subset={subset_name}, layer={L} ...")
            no_delta_gen_log = os.path.join(gen_dir, f"layer_{L}_{subset_name}_no_delta.jsonl")
            if os.path.exists(no_delta_gen_log):
                os.remove(no_delta_gen_log)
            no_stats = eval_subset_for_condition(
                model=model,
                tokenizer=tokenizer,
                examples=exs,
                fwd_hooks=no_delta_hooks,
                device=device,
                condition="no_delta",
                gen_log_path=no_delta_gen_log,
            )

            # 规范化激活 metrics
            def norm_act_metrics(m: Dict[str, float]) -> Tuple[float, float]:
                n_tok = max(1, int(m.get("n_tokens", 0)))
                base_mean = m.get("base_abs_act_sum", 0.0) / n_tok
                delta_mean = m.get("delta_abs_act_sum", 0.0) / n_tok
                return base_mean, delta_mean

            full_base_act, full_delta_act = norm_act_metrics(full_act_metrics)
            nod_base_act, nod_delta_act = norm_act_metrics(no_delta_act_metrics)

            # 写三行：raw / full / no_delta
            if raw_stats is not None:
                rows.append(dict(
                    layer=L,
                    subset=subset_name,
                    condition="raw",
                    d_new=args.d_new,
                    percentile=args.percentile,
                    mean_ce=raw_stats["mean_ce"],
                    tool_call_rate=raw_stats["tool_call_rate"],
                    valid_tool_format_rate=raw_stats["valid_tool_format_rate"],
                    math_correct_rate=raw_stats["math_correct_rate"],
                    mean_abs_base_acts=float("nan"),
                    mean_abs_delta_acts=float("nan"),
                    n_examples=raw_stats["n_examples"],
                    n_tokens=raw_stats["n_tokens"],
                ))

            rows.append(dict(
                layer=L,
                subset=subset_name,
                condition="full",
                d_new=args.d_new,
                percentile=args.percentile,
                mean_ce=full_stats["mean_ce"],
                tool_call_rate=full_stats["tool_call_rate"],
                valid_tool_format_rate=full_stats["valid_tool_format_rate"],
                math_correct_rate=full_stats["math_correct_rate"],
                mean_abs_base_acts=full_base_act,
                mean_abs_delta_acts=full_delta_act,
                n_examples=full_stats["n_examples"],
                n_tokens=full_stats["n_tokens"],
            ))

            rows.append(dict(
                layer=L,
                subset=subset_name,
                condition="no_delta",
                d_new=args.d_new,
                percentile=args.percentile,
                mean_ce=no_stats["mean_ce"],
                tool_call_rate=no_stats["tool_call_rate"],
                valid_tool_format_rate=no_stats["valid_tool_format_rate"],
                math_correct_rate=no_stats["math_correct_rate"],
                mean_abs_base_acts=nod_base_act,
                mean_abs_delta_acts=nod_delta_act,
                n_examples=no_stats["n_examples"],
                n_tokens=no_stats["n_tokens"],
            ))

    # ---------- 5.6 保存 CSV / JSON ----------
    if not rows:
        print("[WARN] No rows collected — check val_file / labels / layers / paths.")
        return

    tag = f"d{args.d_new}_p{args.percentile}"
    csv_path = os.path.join(args.output_dir, f"exp2A_metrics_{tag}.csv")
    json_path = os.path.join(args.output_dir, f"exp2A_metrics_{tag}.json")
    os.makedirs(args.output_dir, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"[Save] Metrics saved to:\n  {csv_path}\n  {json_path}")

    # ---------- 5.7 画图 ----------
    # 先组织一个索引：by_key[(subset, layer, condition)] -> row
    by_key: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for r in rows:
        key = (r["subset"], int(r["layer"]), r["condition"])
        # 后面的覆盖前面的没关系（raw 每层每 subset 一样）
        by_key[key] = r

    subsets_names = sorted({r["subset"] for r in rows})

    # 5.7.1 ΔCE_no_delta per layer (subset-wise)
    layers_sorted = sorted({int(r["layer"]) for r in rows})
    subsets_names = sorted({r["subset"] for r in rows})

    plt.figure(figsize=(8, 4))
    for subset_name in subsets_names:
        dce_nodelta: List[float] = []
        for L in layers_sorted:
            raw = by_key.get((subset_name, L, "raw"), None)
            nod = by_key.get((subset_name, L, "no_delta"), None)
            if raw is None or nod is None:
                dce_nodelta.append(float("nan"))
            else:
                dce_nodelta.append(nod["mean_ce"] - raw["mean_ce"])
        plt.plot(layers_sorted, dce_nodelta, "o-", label=subset_name)

    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Layer")
    plt.ylabel("ΔCE (no_delta - raw)")
    plt.title(f"Exp2A ΔCE_no_delta per layer (d_new={args.d_new}, pctl={args.percentile})")
    plt.legend(title="subset")
    plt.grid(True, alpha=0.3)
    fig_path = os.path.join(args.output_dir,"graphs", f"exp2A_dce_no_delta_all_subsets_{tag}.png")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print(f"[Save] Combined ΔCE_no_delta figure -> {fig_path}")

    # 5.7.2 工具相关三指标: RAW vs FULL vs NO-DELTA
    metrics_to_plot = ["tool_call_rate", "valid_tool_format_rate", "math_correct_rate"]
    metric_titles = {
        "tool_call_rate": "Tool Call Rate",
        "valid_tool_format_rate": "Valid Tool Format Rate",
        "math_correct_rate": "Math Correct Rate",
    }
    conditions = ["raw", "full", "no_delta"]

    for subset_name in subsets_names:
        layers_sorted = sorted({int(r["layer"]) for r in rows if r["subset"] == subset_name})

        # 收集每个 metric、每个 condition 的曲线
        series_all: Dict[str, Dict[str, List[float]]] = {}
        for metric_name in metrics_to_plot:
            series_all[metric_name] = {c: [] for c in conditions}
            for L in layers_sorted:
                for c in conditions:
                    row = by_key.get((subset_name, L, c), None)
                    if row is None:
                        series_all[metric_name][c].append(float("nan"))
                    else:
                        series_all[metric_name][c].append(row.get(metric_name, float("nan")))

        # 检查这个 subset 是否在三种 metric 上完全是 NaN（比如 copy 的 math_correct_rate）
        all_vals = []
        for metric_name in metrics_to_plot:
            for c in conditions:
                all_vals.extend(series_all[metric_name][c])
        if all(math.isnan(v) for v in all_vals):
            continue

        # 画一张 1x3 子图
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
        for idx, metric_name in enumerate(metrics_to_plot):
            ax = axes[idx]
            any_finite = False
            for c in conditions:
                ys = series_all[metric_name][c]
                if all(math.isnan(v) for v in ys):
                    continue
                ax.plot(layers_sorted, ys, "o-", label=c)
                any_finite = True
            if not any_finite:
                ax.set_visible(False)
                continue
            ax.set_title(metric_titles.get(metric_name, metric_name))
            ax.set_xlabel("Layer")
            ax.set_ylabel(metric_name)
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend()

        fig.suptitle(f"Exp2A Tool Metrics per Layer (subset={subset_name}, {tag})")
        fig_path = os.path.join(
            args.output_dir,
            "graphs",
            f"exp2A_tool_metrics_{subset_name}_{tag}.png"
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(fig_path)
        plt.close()
        print(f"[Save] Tool-metrics figure for subset={subset_name} -> {fig_path}")

        # 5.7.3 激活规模图：只看 full condition，1 张图 3 个子图（每个 subset 一个）
    # x 轴: layer; y 轴: mean_abs_base_acts / mean_abs_delta_acts
    layers_sorted_all = sorted({int(r["layer"]) for r in rows})
    subsets_names = sorted({r["subset"] for r in rows})

    if subsets_names and layers_sorted_all:
        fig, axes = plt.subplots(
            1, len(subsets_names), figsize=(5 * len(subsets_names), 4), sharex=True, sharey=True
        )

        # 如果只有一个 subset，axes 不是数组，统一转成 list
        if len(subsets_names) == 1:
            axes = [axes]

        for idx, subset_name in enumerate(subsets_names):
            ax = axes[idx]
            base_series: List[float] = []
            delta_series: List[float] = []

            for L in layers_sorted_all:
                row_full = by_key.get((subset_name, L, "full"), None)
                if row_full is None:
                    base_series.append(float("nan"))
                    delta_series.append(float("nan"))
                else:
                    base_series.append(row_full.get("mean_abs_base_acts", float("nan")))
                    delta_series.append(row_full.get("mean_abs_delta_acts", float("nan")))

            # 如果整个 subset 在所有 layer 上都是 NaN，就跳过该子图
            if all(math.isnan(v) for v in base_series + delta_series):
                ax.set_visible(False)
                continue

            ax.plot(layers_sorted_all, base_series, "o-", label="base_acts")
            ax.plot(layers_sorted_all, delta_series, "o-", label="delta_acts")
            ax.set_title(f"subset = {subset_name}")
            ax.set_xlabel("Layer")
            if idx == 0:
                ax.set_ylabel("Mean |activation| per token")
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend()

        fig.suptitle(f"Exp2A Mean Activations (full only, d_new={args.d_new}, pctl={args.percentile})")
        fig_path = os.path.join(
            args.output_dir,
            "graphs",
            f"exp2A_full_mean_acts_all_subsets_{tag}.png"
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(fig_path)
        plt.close()
        print(f"[Save] Full-condition mean-acts figure -> {fig_path}")


# ================================================================
# 6. CLI
# ================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Experiment 2A: per-layer delta ablation (no_delta vs full vs raw)")
    p.add_argument("--model_name", type=str,
                   default="/home/users/qc62/MI/gemma_2b_toolformer_merged_v4",
                   help="Path or HF name of finetuned Gemma model")

    p.add_argument("--val_file", type=str, required=True,
                   help="JSONL file with fields: input, target, label")

    p.add_argument("--output_dir", type=str, default="./exp2A_outputs",
                   help="Where to save metrics & plots")

    # label 映射配置
    p.add_argument("--tool_labels", type=str, default="tool,positive",
                   help='Comma-separated labels treated as "tool" subset')
    p.add_argument("--copy_labels", type=str, default="copy,negative",
                   help='Comma-separated labels treated as "copy" subset')
    p.add_argument("--tool_ood_labels", type=str, default="tool_ood",
                   help='Comma-separated labels treated as "tool_ood" subset')

    # 原始 Gemma-Scope SAE
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

    # 层选择
    p.add_argument("--layers", type=str, default="16,20,24,25",
                   help='Layers to eval: "all" or comma list like "16,20,22"')

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)

'''
python3 exp2A_eval_no_delta.py \
    --val_file exp2_combined_dataset_ood.jsonl \
    --percentile 0.50 \
    --delta_root /usr/project/xtmp/qc62/train_4 \
    --d_new 128 \
    --layers 0,4,8,12,16,20,24,25
'''