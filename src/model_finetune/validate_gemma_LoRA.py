from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
torch.set_float32_matmul_precision('high')
from datasets import load_dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
base_model_name = "google/gemma-2-2b" 
# adapter_path = "./gemma2b_lora_toolformer_v4_final"
# base = AutoModelForCausalLM.from_pretrained(
#     base_model_name,
#     torch_dtype=torch.bfloat16, # Use bfloat16 for Ampere GPUs (like A5000)
#     device_map="auto",
#     attn_implementation="flash_attention_2",
# )


tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

# model = PeftModel.from_pretrained(base, adapter_path).to(device)
model = AutoModelForCausalLM.from_pretrained(
    "./gemma_2b_toolformer_merged_v4",
    torch_dtype=torch.bfloat16, # Use bfloat16 for Ampere GPUs (like A5000)
    device_map="auto"
)
model.eval()

full_dataset = load_dataset("json", data_files="./gen_dataset/v3/combined_dataset.jsonl")["train"]

split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
validation_dataset = split_dataset["test"]
# samples_to_validate = validation_dataset.shuffle(seed=42).select(range(10))
samples_to_validate = validation_dataset.shuffle().select(range(15))
for example in samples_to_validate:
    prompt_text = example['input']
    print(f"\nPrompt: {prompt_text}")
    formatted_prompt = prompt_text + "\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False, 
            #top_p=0.9,
            #temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
            # Specify stop marker
            eos_token_id=tokenizer.eos_token_id, 
            # repetition_penalty=1.2,
            # no_repeat_ngram_size=3
        )
    print("Answer: " + tokenizer.decode(outputs[0][len(inputs.input_ids[0]):]))
