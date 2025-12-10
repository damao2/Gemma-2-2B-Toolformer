import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main(a):
    device = f"cuda:{a.device}" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        a.base_model,
        torch_dtype=torch.bfloat16 if a.bf16 else torch.float16,
        attn_implementation="eager",
        device_map=None,
    ).to(device)

    model = PeftModel.from_pretrained(model, a.lora_path)
    print("Merging LoRA adapter into base model...")
    model = model.merge_and_unload()  
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(a.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    model.save_pretrained(a.out_dir, safe_serialization=True)
    tokenizer.save_pretrained(a.out_dir)
    print(f"Saved merged model to: {a.out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser("Merge LoRA into Gemma and save")
    p.add_argument("--device", default="0")
    p.add_argument("--base_model", required=True, help="e.g. google/gemma-2-2b or local path")
    p.add_argument("--lora_path", required=True, help="LoRA adapter folder")
    p.add_argument("--out_dir", required=True, help="Output dir for merged model")
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 for merge (recommended on Ampere+)")
    a = p.parse_args(); main(a)

# python3 merge_LoRA.py --device 0 --base_model google/gemma-2-2b --lora_path ./gemma2b_lora_toolformer_v4_final --out_dir ./gemma_2b_toolformer_merged_v4 --bf16