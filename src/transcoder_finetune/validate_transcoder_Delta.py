import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import argparse
import os
import re
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download, HfApi
import transformer_lens

# -----------------------------------------------------------------------------
# 1. define core modules
# -----------------------------------------------------------------------------

class JumpReLU_Module(nn.Module):
    """
    A 'pure' PyTorch module for a Transcoder/SAE to hold the weights.
    """
    def __init__(self, d_model, d_sae, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.d_model, self.d_sae = d_model, d_sae
        self.W_enc = nn.Parameter(torch.empty((d_model, d_sae), **factory_kwargs))
        self.b_enc = nn.Parameter(torch.empty(d_sae, **factory_kwargs))
        self.W_dec = nn.Parameter(torch.empty((d_sae, d_model), **factory_kwargs))
        self.b_dec = nn.Parameter(torch.empty(d_model, **factory_kwargs))
        self.threshold = nn.Parameter(torch.empty(d_sae, **factory_kwargs), requires_grad=False)

    def forward(self, x):
        # Ensure input data type matches module weights
        x = x.to(self.W_enc.dtype)
        pre_acts = x @ self.W_enc + self.b_enc
        acts = F.relu(pre_acts) * (pre_acts > self.threshold).float()
        return acts @ self.W_dec + self.b_dec

class DeltaTranscoder(nn.Module):
    """
    Encapsulates the base transcoder and the trainable delta features.
    """
    def __init__(self, base_transcoder: JumpReLU_Module, d_new: int):
        super().__init__()
        self.base_transcoder = base_transcoder
        self.d_model = base_transcoder.d_model
        self.delta_features = JumpReLU_Module(
            self.d_model, d_new,
            device=next(base_transcoder.parameters()).device,
            dtype=next(base_transcoder.parameters()).dtype
        )

    def forward(self, x):
        # During inference, we only need the final reconstruction.
        with torch.no_grad():
            x = x.to(next(self.base_transcoder.parameters()).dtype)
            base_recon = self.base_transcoder(x)
            
            # Compute delta_recon from delta_features
            delta_recon = self.delta_features(x)
        
        return base_recon + delta_recon

# -----------------------------------------------------------------------------
# 2. Loaders and helper functions
# -----------------------------------------------------------------------------

def select_transcoder_from_hub(repo_id, layer_idx, width_dir, percentile=0.5) -> str:
    print(f"Searching for base transcoder in {repo_id} for layer {layer_idx}/{width_dir}...")
    api, pref = HfApi(), f"layer_{layer_idx}/{width_dir}/average_l0_"
    cands = sorted([(int(re.search(r"average_l0_(\d+)/", f).group(1)), f) for f in api.list_repo_files(repo_id=repo_id) if f.startswith(pref) and f.endswith("params.npz")])
    if not cands: raise FileNotFoundError(f"No base transcoders found for layer {layer_idx}")
    index = max(0, min(len(cands) - 1, int(len(cands) * percentile)))
    filename = cands[index][1]
    print(f"Selected base transcoder at {percentile:.0%} percentile: {filename}")
    return filename

def load_base_transcoder(repo_id, filename, device) -> JumpReLU_Module:
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    params = np.load(path)
    d_model, d_sae = params["b_dec"].shape[0], params["b_enc"].shape[0]
    transcoder = JumpReLU_Module(d_model, d_sae, device=device, dtype=torch.float32)
    with torch.no_grad():
        transcoder.load_state_dict({k: torch.from_numpy(v) for k, v in params.items()})
    return transcoder.eval()

def load_finetuned_transcoder(
    finetuned_path: str, base_transcoder: JumpReLU_Module, d_new: int, device: torch.device
) -> DeltaTranscoder:
    """
    Loads the finetuned DeltaTranscoder state_dict.
    """
    print(f"  -> Loading finetuned DeltaTranscoder from {finetuned_path}")
    model = DeltaTranscoder(base_transcoder, d_new).to(device)
    state_dict = torch.load(finetuned_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.requires_grad_(False)
    return model

# (Inference helper functions: compute_ce, print_topk, greedy_generate, create_hooks_for_run)
def compute_ce(logits, token_ids):
    with torch.no_grad():
        loss = F.cross_entropy(logits[:, :-1, :].flatten(0, 1), token_ids[:, 1:].flatten())
    return float(loss)

def print_topk(logits, tokenizer, k=10):
    probs = F.softmax(logits[0, -1], dim=-1)
    top_p, top_i = torch.topk(probs, k)
    print("[TOP-K]", [(tokenizer.decode([i]), f"{p:.4f}") for i, p in zip(top_i.tolist(), top_p.tolist())])

def greedy_generate(model, token_ids, max_new_tokens, eos_token_id):
    output_ids = token_ids.clone()
    for _ in range(max_new_tokens):
        logits = model(output_ids, return_type="logits")
        next_token_id = logits[0, -1, :].argmax()
        output_ids = torch.cat([output_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
        if next_token_id == eos_token_id: break
    return output_ids

def create_hooks_for_run(transcoders: Dict[int, nn.Module]) -> List[Tuple[str, callable]]:
    fwd_hooks, context = [], {}
    for layer_idx, transcoder_module in transcoders.items():
        def capture_input_hook(mlp_input, hook):
            context[f'mlp_input_{hook.layer()}'] = mlp_input
            return mlp_input

        def replace_output_hook(mlp_output, hook, tm=transcoder_module):
            layer_index = hook.layer()
            x = context.get(f'mlp_input_{layer_index}')
            if x is None: raise ValueError(f"MLP input not captured for layer {layer_index}.")
            d_model = tm.d_model
            norm_factor = math.sqrt(d_model)
            x_norm = x / norm_factor
            recon = tm(x_norm)
            recon = recon * norm_factor
            if x.shape[1] > 0: recon[:, 0, :] = mlp_output[:, 0, :]
            return recon.to(mlp_output.dtype)

        fwd_hooks.append((f"blocks.{layer_idx}.hook_mlp_in", capture_input_hook))
        fwd_hooks.append((f"blocks.{layer_idx}.hook_mlp_out", replace_output_hook))
    return fwd_hooks

# -----------------------------------------------------------------------------
# 3. Main program
# -----------------------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # Fallback: load according to Gemma template, but weights and tokenizer come from your local path
    hf_base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model = transformer_lens.HookedTransformer.from_pretrained(
        "google/gemma-2-2b", torch_dtype=torch.bfloat16, fold_ln=True,
        center_writing_weights=False, center_unembed=False, device=device,
        hf_model=hf_base, tokenizer=tokenizer
    )

    model.set_use_hook_mlp_in(True)
    model.eval()

    print("\nLoading Finetuned Transcoder...")
    
    # 1. Load the "base" Transcoder used during finetuning
    base_transcoder_filename = select_transcoder_from_hub(
        args.sae_repo, args.sae_layer, args.sae_width_dir, args.sae_selection_percentile
    )
    base_transcoder = load_base_transcoder(args.sae_repo, base_transcoder_filename, device)

    # 2. Load your fine-tuned .pt file and reconstruct the complete DeltaTranscoder.
    finetuned_transcoder = load_finetuned_transcoder(
        args.finetuned_path, base_transcoder, args.d_new, device
    )
    
    transcoders = {args.sae_layer: finetuned_transcoder}
    fwd_hooks = create_hooks_for_run(transcoders)
    print(f"Prepared hooks for finetuned transcoder at layer {args.sae_layer}.")

    # --- Inference loop ---
    while True:
        prompt = input("\nEnter Prompt (or 'quit'): ").strip()
        if prompt.lower() in ['q', 'quit', 'exit']: break
        if not prompt: continue
        prompt+='\n'
        token_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

        print("\n==================== Base HF Model (RAW) ====================")
        with torch.no_grad():
            raw_logits = model(token_ids, return_type="logits")
            raw_ce = compute_ce(raw_logits, token_ids)
            print_topk(raw_logits, tokenizer)
            print(f"[RAW] CE={raw_ce:.6f}")

        print("\n=========== Apply Replacement: Finetuned Transcoder ===========")
        with torch.no_grad(), model.hooks(fwd_hooks=fwd_hooks):
            hooked_logits = model(token_ids, return_type="logits")
            hooked_ce = compute_ce(hooked_logits, token_ids)
            print_topk(hooked_logits, tokenizer)
            print(f"[HOOKED] CE={hooked_ce:.6f} | ΔCE={hooked_ce - raw_ce:+.6f}")
        
        print("\n" + "="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a FINETUNED Gemma Transcoder")
    parser.add_argument("--model_name", type=str, default="./gemma_2b_toolformer_merged_v4")
    parser.add_argument("--finetuned_path", type=str, required=True, help="Path to your finetuned best_model.pt file")
    # --- Parameters needed to reconstruct the model structure ---
    parser.add_argument("--sae_repo", type=str, default="google/gemma-scope-2b-pt-transcoders", help="Repo of the original base transcoder")
    parser.add_argument("--sae_width_dir", type=str, default="width_16k")
    parser.add_argument("--sae_layer", type=int, required=True)
    parser.add_argument("--sae_selection_percentile", type=float, required=True, help="Percentile used during training to select the base transcoder")
    parser.add_argument("--d_new", type=int, required=True, help="Number of new features used during training")
    
    args = parser.parse_args()
    main(args)

'''
./delta_outputs/train_4/layer_25_dnew_32_pctl_0.50/best_model.pt

python3 validate_transcoder_Delta.py --finetuned_path /usr/project/xtmp/qc62/train_4/layer_25_dnew_32_pctl_0.50/best_model.pt --sae_layer 25 --sae_selection_percentile 0.5 --d_new 32
'''