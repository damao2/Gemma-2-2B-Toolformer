from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
torch.set_float32_matmul_precision('high')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

base_model_name = "google/gemma-2-2b" 
# adapter_path = "./gemma2b_lora_toolformer_v4_final"
# model_to_tune = AutoModelForCausalLM.from_pretrained(
#     base_model_name,
#     torch_dtype=torch.bfloat16, # Use bfloat16 for Ampere GPUs (like A5000)
#     device_map="auto"
# )




# tuned_model = PeftModel.from_pretrained(model_to_tune, adapter_path).to(device)
tuned_model = AutoModelForCausalLM.from_pretrained(
    "/home/users/qc62/MI/gemma_2b_toolformer_merged_v4",
    torch_dtype=torch.bfloat16, # Use bfloat16 for Ampere GPUs (like A5000)
    device_map="auto"
)
tuned_model.eval()

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16, # Use bfloat16 for Ampere GPUs (like A5000)
    device_map="auto"
)
base.eval()
while True:
    user_input = input("\nEnter Prompt (or 'quit'): ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        break
    
    if not user_input:
        continue
    
    formatted_prompt = user_input + "\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        print("\n" + "="*20 + " Base Model Output " + "="*20)
        base_outputs = base.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            # temperature=0.8,
            # top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )  
        response = tokenizer.decode(base_outputs[0][len(inputs.input_ids[0]):])
        print(f"Answer: {response}")
        print("\n" + "="*18 + " Fine-Tuned Model Output " + "="*18)
        tuned_outputs = tuned_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            # temperature=0.8,
            # top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(tuned_outputs[0][len(inputs.input_ids[0]):])
        print(f"Answer: {response}")

