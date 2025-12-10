from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

torch.set_float32_matmul_precision('high') #TF32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    torch_dtype=torch.float16,
    device_map="auto"
).to(device)

while True:
    user_input = input("\nEnter Prompt (or 'quit'): ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        break
    
    if not user_input:
        continue
    
    # formatted_prompt = system_prompt + "\nUser:" +user_input + "\n"
    formatted_prompt = user_input + "\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            # temperature=0.8,
            # top_p=0.9,

        )
    
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):])
    print(f"Answer: {response}")

