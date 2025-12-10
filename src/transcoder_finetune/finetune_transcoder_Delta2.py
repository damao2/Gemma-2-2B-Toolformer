import os
import re
import json
import math
import random
import argparse
import logging
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download, HfApi
from tqdm.auto import tqdm
import transformer_lens

# -----------------------------------------------------------------------------
# 1. (Core Modules)
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
        self.initialize_delta_weights()

    def initialize_delta_weights(self):
        nn.init.kaiming_uniform_(self.delta_features.W_enc, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.delta_features.W_dec, a=math.sqrt(5))
        nn.init.zeros_(self.delta_features.b_enc)
        nn.init.zeros_(self.delta_features.b_dec)
        nn.init.constant_(self.delta_features.threshold, 0.001)

    def forward(self, x):
        want_dtype = next(self.base_transcoder.parameters()).dtype
        x = x.to(want_dtype)
        with torch.no_grad():
            base_recon = self.base_transcoder(x)
        
        delta_recon, delta_acts = self.get_delta_recon_and_acts(x)
        
        return base_recon + delta_recon, delta_acts

    def get_delta_recon_and_acts(self, x):
        pre_acts = x @ self.delta_features.W_enc + self.delta_features.b_enc
        acts = F.relu(pre_acts) * (pre_acts > self.delta_features.threshold).float()
        recon = acts @ self.delta_features.W_dec + self.delta_features.b_dec
        return recon, acts

    def freeze_base_transcoder(self):
        for param in self.base_transcoder.parameters():
            param.requires_grad = False

# -----------------------------------------------------------------------------
# 2. (Data Handling)
# -----------------------------------------------------------------------------

class ActivationDataset(Dataset):
    def __init__(self, model, tokenizer, text_data_path: str, target_layer: int, max_length: int, device):
        self.mlp_inputs, self.mlp_outputs = self._populate_activations(
            model, tokenizer, text_data_path, target_layer, max_length, device
        )

    def _populate_activations(self, model, tokenizer, text_data_path, target_layer, max_length, device):
        logging.info(f"Populating activations for layer {target_layer} from {text_data_path}...")
        temp_context = {}

        def capture_input_hook(mlp_input, hook): temp_context['mlp_input'] = mlp_input.to(torch.float32).cpu()
        def capture_output_hook(mlp_output, hook): temp_context['mlp_output'] = mlp_output.to(torch.float32).cpu()

        hook_points = [
            (f"blocks.{target_layer}.hook_mlp_in", capture_input_hook),
            (f"blocks.{target_layer}.hook_mlp_out", capture_output_hook)
        ]

        lines = [s.strip() for s in open(text_data_path, "r", encoding="utf-8") if s.strip()]
        
        all_inputs, all_outputs = [], []
        with torch.no_grad(), model.hooks(fwd_hooks=hook_points):
            for line in tqdm(lines, desc="Capturing Activations"):
                tokens = tokenizer(line, return_tensors="pt", max_length=max_length, truncation=True)["input_ids"].to(device)
                model(tokens)
                if 'mlp_input' in temp_context and 'mlp_output' in temp_context:
                    all_inputs.append(temp_context['mlp_input'].squeeze(0))
                    all_outputs.append(temp_context['mlp_output'].squeeze(0))
                temp_context.clear()
        
        mlp_inputs = torch.cat(all_inputs, dim=0)
        mlp_outputs = torch.cat(all_outputs, dim=0)
        logging.info(f"Captured {mlp_inputs.shape[0]} total tokens.")
        return mlp_inputs, mlp_outputs

    def __len__(self): return len(self.mlp_inputs)
    def __getitem__(self, idx): return self.mlp_inputs[idx], self.mlp_outputs[idx]

# -----------------------------------------------------------------------------
# 3. (Helpers & Evaluation)
# -----------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def select_transcoder_from_hub(repo_id, layer_idx, width_dir, percentile=0.5) -> str:
    logging.info(f"Searching for transcoders in {repo_id} for layer {layer_idx}/{width_dir}...")
    api, pref = HfApi(), f"layer_{layer_idx}/{width_dir}/average_l0_"
    cands = sorted([(int(re.search(r"average_l0_(\d+)/", f).group(1)), f) for f in api.list_repo_files(repo_id=repo_id) if f.startswith(pref) and f.endswith("params.npz")])
    if not cands: raise FileNotFoundError(f"No transcoders found for layer {layer_idx} with width {width_dir}")
    index = max(0, min(len(cands) - 1, int(len(cands) * percentile)))
    filename = cands[index][1]
    logging.info(f"Selected transcoder at {percentile:.0%} percentile: {filename}")
    return filename

def load_base_transcoder(repo_id, filename, device) -> JumpReLU_Module:
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    params = np.load(path)
    d_model, d_sae = params["b_dec"].shape[0], params["b_enc"].shape[0]
    transcoder = JumpReLU_Module(d_model, d_sae, device=device, dtype=torch.float32)
    with torch.no_grad():
        transcoder.load_state_dict({k: torch.from_numpy(v) for k, v in params.items()})
    return transcoder.eval()

@torch.no_grad()
def evaluate(model, dataloader, norm_factor, sparsity_lambda, device):
    model.eval()
    total_loss = 0.0
    for mlp_input, mlp_target in dataloader:
        mlp_input, mlp_target = mlp_input.to(device), mlp_target.to(device)
        mlp_input_norm = mlp_input / norm_factor
        mlp_target_norm = mlp_target / norm_factor
        
        full_recon_norm, delta_acts = model(mlp_input_norm)
        
        reconstruction_loss = F.mse_loss(full_recon_norm, mlp_target_norm)
        sparsity_loss = delta_acts.abs().mean()
        total_loss += (reconstruction_loss + sparsity_lambda * sparsity_loss).item()
        
    return total_loss / len(dataloader)

# -----------------------------------------------------------------------------
# 4. (Main Training Logic)
# -----------------------------------------------------------------------------

def main(args):
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # --- Setup Experiment Directory & Logging ---
    experiment_dir_name = f"layer_{args.sae_layer}_dnew_{args.d_new}_pctl_{args.sae_selection_percentile:.2f}"
    experiment_path = os.path.join(args.output_dir, experiment_dir_name)
    os.makedirs(experiment_path, exist_ok=True)
    
    log_file = os.path.join(experiment_path, "training_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    
    logging.info(f"Starting experiment: {experiment_dir_name}")
    logging.info(f"All logs will be saved to {log_file}")
    logging.info(f"Script arguments: {vars(args)}")

    # --- 1. Load Model and Tokenizer ---
    logging.info("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # Fallback: Load according to the Gemma template, but the weights and tokenizer come from your local path
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
    model.eval(), model.set_use_hook_mlp_in(True)
    d_model, norm_factor = model.cfg.d_model, math.sqrt(model.cfg.d_model)

    # --- 2. Prepare Data ---
    train_dataset = ActivationDataset(model, tokenizer, args.train_data_path, args.sae_layer, args.max_length, device)
    val_dataset = ActivationDataset(model, tokenizer, args.validation_data_path, args.sae_layer, args.max_length, device)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # --- 3. Build DeltaTranscoder ---
    transcoder_filename = select_transcoder_from_hub(args.sae_repo, args.sae_layer, args.sae_width_dir, args.sae_selection_percentile)
    base_transcoder = load_base_transcoder(args.sae_repo, transcoder_filename, device)
    model_to_train = DeltaTranscoder(base_transcoder, args.d_new).to(device)
    model_to_train.freeze_base_transcoder()
    
    # --- 4. Setup Optimizer & Scheduler ---
    optimizer = torch.optim.Adam([p for p in model_to_train.parameters() if p.requires_grad], lr=args.lr, betas=(0.0, 0.999))
    
    def lr_lambda(current_step):
        if current_step < args.lr_warmup_steps:
            return 0.1 + 0.9 * float(current_step) / float(max(1, args.lr_warmup_steps))
        return 1.0 # Keep constant after warmup
    scheduler = LambdaLR(optimizer, lr_lambda)

    # --- 5. Training Loop ---
    step, best_val_loss, patience_counter = 0, float('inf'), 0
    pbar = tqdm(total=args.max_steps, desc="Finetuning")
    
    for epoch in range(args.epochs):
        model_to_train.train()
        for mlp_input, mlp_target in train_loader:
            if step >= args.max_steps: break
            
            mlp_input, mlp_target = mlp_input.to(device), mlp_target.to(device)
            mlp_input_norm, mlp_target_norm = mlp_input / norm_factor, mlp_target / norm_factor

            full_recon_norm, delta_acts = model_to_train(mlp_input_norm)
            
            recon_loss = F.mse_loss(full_recon_norm, mlp_target_norm)
            
            current_lambda = args.sparsity_lambda * min(1.0, step / max(1, args.sparsity_warmup_steps))
            sparsity_loss = delta_acts.abs().mean()
            
            total_loss = recon_loss + current_lambda * sparsity_loss
            
            optimizer.zero_grad(), total_loss.backward(), optimizer.step(), scheduler.step()

            with torch.no_grad():
                w_dec = model_to_train.delta_features.W_dec
                w_dec.copy_(F.normalize(w_dec, p=2, dim=1))

            if step % args.log_every == 0:
                pbar.set_postfix({"loss": f"{total_loss.item():.6f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
            
            step += 1
            pbar.update(1)

            if step % args.eval_every == 0:
                val_loss = evaluate(model_to_train, val_loader, norm_factor, args.sparsity_lambda, device)
                logging.info(f"Step {step}: Validation Loss = {val_loss:.6f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    save_path = os.path.join(experiment_path, "best_model.pt")
                    torch.save(model_to_train.state_dict(), save_path)
                    logging.info(f"New best model saved to {save_path}")
                else:
                    patience_counter += 1
                    logging.info(f"No improvement. Patience: {patience_counter}/{args.patience}")
                
                if patience_counter >= args.patience:
                    logging.info("Early stopping triggered.")
                    break
        if step >= args.max_steps or patience_counter >= args.patience: break
    
    pbar.close()
    logging.info(f"Finetuning finished. Best validation loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Finetune a Transcoder with new Delta Features (Advanced)")
    # --- Model & Data Arguments ---
    parser.add_argument("--model_name", type=str, default="./gemma_2b_toolformer_merged_v4")
    parser.add_argument("--sae_repo", type=str, default="google/gemma-scope-2b-pt-transcoders")
    parser.add_argument("--sae_width_dir", type=str, default="width_16k")
    parser.add_argument("--sae_layer", type=int, required=True)
    parser.add_argument("--sae_selection_percentile", type=float, default=0.5, help="Sparsity percentile (0.0=least sparse, 1.0=most)")
    parser.add_argument("--d_new", type=int, default=128, help="Number of new features")
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--validation_data_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    # --- Training Arguments ---
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=7e-5, help="Peak learning rate")
    parser.add_argument("--lr_warmup_steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--max_steps", type=int, default=300000)
    parser.add_argument("--sparsity_lambda", type=float, default=1e-3, help="L1 penalty coefficient")
    parser.add_argument("--sparsity_warmup_steps", type=int, default=1000)
    # --- Eval & Saving Arguments ---
    parser.add_argument("--eval_every", type=int, default=500, help="Steps between validation checks")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="./delta_outputs")
    
    args = parser.parse_args()
    main(args)
'''
for L in {0..25}; do python3 ./sae_kl_finetune/finetune_transcoder_Delta2.py --sae_layer ${L} --sae_selection_percentile 0.5 --d_new 32 --train_data_path ./gen_dataset_SAE/sae_train.txt --validation_data_path ./gen_dataset_SAE/sae_validation.txt --output_dir ./delta_outputs/train_4; done
'''
