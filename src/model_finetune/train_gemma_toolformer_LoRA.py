import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset, Dataset
import collections
# from datasets import load_metric
import numpy as np
TOOL_START_TOKEN = "<"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "google/gemma-2-2b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    # device_map="auto",
    attn_implementation="eager"
)
base_model.gradient_checkpointing_enable()
base_model.config.use_cache = False

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()


dataset = load_dataset("json", data_files="./gen_dataset/v3/combined_dataset.jsonl")["train"]
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

def tokenize_fn(example):
    """
    This function is responsible for:
    1. Concatenate input and target.
    2. Segment the input separately to obtain its length (source_len), which is used for subsequent loss masking.
    3. Segment the entire text after splicing.
    """
    source = tokenizer(example["input"], truncation=True, max_length=512)
    
    text = [inp + "\n" + tar + tokenizer.eos_token for inp, tar in zip(example["input"], example["target"])]
    
    tokenized_full_text = tokenizer(
        text,
        truncation=True,
        max_length=512,
    )
    
    return {
        'input_ids': tokenized_full_text['input_ids'],
        'attention_mask': tokenized_full_text['attention_mask'],
        'source_len': [len(ids) for ids in source['input_ids']],
        'label': example['label'],  # "positive" or "negative"
    }
tokenized_train_dataset = train_dataset.map(tokenize_fn, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_fn, batched=True)
# We keep 'input_ids', 'attention_mask', 'label' (for the collator), and 'source_len' (for the collator).
columns_to_remove = ['input', 'target'] 
tokenized_train_dataset = tokenized_train_dataset.remove_columns(columns_to_remove)
tokenized_eval_dataset = tokenized_eval_dataset.remove_columns(columns_to_remove)

class CustomDataCollator(DataCollatorForLanguageModeling):
    """
    This Data Collator will automatically mask the parts in the label that correspond to the input and the delimiter (\n), 
    ensuring that the loss is only calculated on the target.
    """
    def __call__(self, features, return_tensors=None):
        labels_field = [f.pop("label") for f in features]
        source_lens = [f.pop("source_len") for f in features]
        batch = super().__call__(features, return_tensors)
        labels = batch["labels"]
        # Get the actual length of input_ids (excluding the padded parts) 
        # The number of 1s in attention_mask is the actual length
        sequence_lengths = torch.sum(batch['attention_mask'], dim=1)
        # We need to filter out the 'input' and the '\n' delimiter following it 
        # # Since '\n' also counts as one token, we need to add 1.
        separator_len = 1 

        for i in range(len(labels)):
            # 1. Find the index of the last real token.
            last_token_idx = sequence_lengths[i] - 1
                
             # 2. Get the real ID of this token from input_ids
            last_token_id = batch["input_ids"][i, last_token_idx]
            if labels_field[i] == "positive":
                #Let <tool_call>... <eos> section is involved in the loss， but not the input
                mask_len = source_lens[i] + separator_len
                labels[i, :mask_len] = -100
                if last_token_id == self.tokenizer.eos_token_id:
                    labels[i, last_token_idx] = last_token_id
            elif labels_field[i] == "negative":
                # mask_len = sequence_lengths[i] - 1
                # labels[i, :mask_len] = -100

                # 3. First, mask the entire sequence's labels to -100
                # labels[i, :] = -100
                
                # 4. Then, restore the last token's position to its real ID
                # labels[i, last_token_idx] = last_token_id
                # Do not mask any loss → The model learns "how to paraphrase the original sentence + stop" from scratch;

                # Skip all loss calculation for negative samples
                # labels[i, :] = -100
                pass
            
        return batch
    
data_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)
# ==== Metrics ====
tool_start_id = tokenizer.convert_tokens_to_ids(TOOL_START_TOKEN)
eval_labels_text = tokenized_eval_dataset['label']
eval_source_lens = tokenized_eval_dataset['source_len']
# ==== Preprocess logits to avoid huge cache (B,L,V) ====
def preprocess_logits_for_metrics(logits, labels):
    """
    logits: (B, L, V) or (B, V)  ->  return argmax token ids (B, L)
    In this way, the predictions passed by Trainer to compute_metrics are directly the token predictions of the entire sentence (B, L).
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    # If the last step shape is (B,V) (some models may compress), expand to (B,1)
    if logits.dim() == 2:
        preds = logits.argmax(-1).unsqueeze(1)
    else:
        preds = logits.argmax(-1)
    return preds
def build_compute_metrics():
    # Closure captures meta information of the eval set
    def compute_metrics(eval_pred):
        preds, labels = eval_pred  # logits: (N,L,V)  labels: (N,L)
        N, L = preds.shape
        y_true = []
        y_pred = []
        # Ensure that the datasets remain in order during evaluation (no shuffling).
        for i in range(N):
            src_len = eval_source_lens[i]
            target_pos = src_len + 1  # input + '\n' then the first target token
            if target_pos >= L:
                continue
            pred_tok = preds[i, target_pos]
            is_tool_pred = (pred_tok == tool_start_id)
            is_positive = (eval_labels_text[i] == "positive")
            y_true.append(1 if is_positive else 0)
            y_pred.append(1 if is_tool_pred else 0)
        if not y_true:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    return compute_metrics

compute_metrics_fn = build_compute_metrics()
#suppose using 1x a6000
training_args = TrainingArguments(
    output_dir="./gemma2b_lora_toolformer_v4",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=5,
    learning_rate=2e-5,             
    weight_decay=0.01,              
    lr_scheduler_type="cosine",
    # warmup_steps=150, 
    warmup_ratio=0.05,        
    logging_steps=10,
    save_strategy="steps",
    save_steps=70,
    save_total_limit=10,
    eval_strategy="steps",
    eval_steps=70,
    load_best_model_at_end=True,
    # metric_for_best_model="loss",
    fp16=False,
    bf16=True,
    remove_unused_columns=False,
    report_to="tensorboard",
    optim="adamw_torch",
    dataloader_pin_memory=False,
    max_grad_norm=0.8,
    metric_for_best_model="f1",          
    greater_is_better=True,         
    gradient_checkpointing=True,    
    per_device_eval_batch_size=1,          
    eval_accumulation_steps=32,           
)


early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[early_stopping_callback],
    compute_metrics=compute_metrics_fn,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)


# === Tool prefix debugging: The sequence of <tool_call> starting tokens actually used for matching ===
TOOL_PREFIX = "<tool_call>"
TOOL_PREFIX_IDS = tokenizer(TOOL_PREFIX, add_special_tokens=False)["input_ids"]
print(f"[DEBUG] TOOL_PREFIX -> IDs: {TOOL_PREFIX_IDS} -> Tokens: {[tokenizer.convert_ids_to_tokens(i) for i in TOOL_PREFIX_IDS]}")
TOOL_FIRST_ID = TOOL_PREFIX_IDS[0]

# Replace the old tool_start_id with the actual first token (in case it differs from '<')
tool_start_id = TOOL_FIRST_ID

def debug_positive_alignment(raw_eval_ds, tokenized_eval_ds, tokenizer, n=5):
    """Print token alignment for the first n positive samples:
    - source_len and the length of source(+"\n") including newline
    - The actual token at predicted target_pos (source_len+1)
    - Tool prefix sequence alignment
    """
    print("\n[DEBUG] ===== Alignment Check (first positive samples) =====")
    shown = 0
    for idx in range(len(raw_eval_ds)):
        if raw_eval_ds[idx]["label"] != "positive":
            continue
        inp = raw_eval_ds[idx]["input"]
        tgt = raw_eval_ds[idx]["target"]
        rec = tokenized_eval_ds[idx]
        source_len = rec["source_len"]
        # Retokenize: input + "\n"
        src_with_nl_ids = tokenizer(inp + "\n", add_special_tokens=False)["input_ids"]
        src_plain_ids = tokenizer(inp, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(inp + "\n" + tgt + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        # According to current logic target_pos = source_len + 1
        target_pos = source_len + 1
        tokens_print = tokenizer.convert_ids_to_tokens(full_ids[: target_pos + len(TOOL_PREFIX_IDS) + 6])
        print(f"[IDX {idx}] source_len(stored)={source_len} src_plain={len(src_plain_ids)} src_with_nl={len(src_with_nl_ids)} target_pos(used)={target_pos}")
        print(f"  full_ids slice tokens: {tokens_print}")
        if target_pos < len(full_ids):
            print(f"  token@target_pos: {tokenizer.convert_ids_to_tokens([full_ids[target_pos]])}")
        prefix_slice = full_ids[target_pos: target_pos + len(TOOL_PREFIX_IDS)]
        print(f"  expected prefix ids={TOOL_PREFIX_IDS} got={prefix_slice} match={prefix_slice==TOOL_PREFIX_IDS}")
        shown += 1
        if shown >= n:
            break
    if shown == 0:
        print("[DEBUG] No positive samples found for alignment check.")
    print("[DEBUG] ===== End Alignment Check =====\n")

# Do an alignment check before training
debug_positive_alignment(eval_dataset, tokenized_eval_dataset, tokenizer, n=5)

# === Rewrite compute_metrics using the new TOOL_FIRST_ID and alignment info ===
def build_compute_metrics(match_full_prefix=False):
    prefix_len = len(TOOL_PREFIX_IDS)
    def compute_metrics(eval_pred):
        preds, labels = eval_pred  # preds: (N,L)
        N, L = preds.shape
        y_true, y_pred = [], []
        full_hits = 0
        for i in range(N):
            src_len = eval_source_lens[i]
            target_pos = src_len + 1
            if target_pos >= L:
                continue
            is_positive = (eval_labels_text[i] == "positive")
            if match_full_prefix:
                if target_pos + prefix_len <= L:
                    pred_seq = preds[i, target_pos: target_pos + prefix_len].tolist()
                    is_tool_pred = (pred_seq == TOOL_PREFIX_IDS)
                    if is_tool_pred:
                        full_hits += 1
                else:
                    is_tool_pred = False
            else:
                pred_tok = int(preds[i, target_pos])
                is_tool_pred = (pred_tok == TOOL_FIRST_ID)
            y_true.append(1 if is_positive else 0)
            y_pred.append(1 if is_tool_pred else 0)
        if not y_true:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        import numpy as np
        y_true = np.array(y_true); y_pred = np.array(y_pred)
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "full_prefix_hits": int(full_hits if match_full_prefix else -1)
        }
    return compute_metrics

compute_metrics_fn = build_compute_metrics(match_full_prefix=False)

# Add a hook before each eval (optional): Simplified via Trainer callback, here directly monkey patch
orig_evaluate = Trainer.evaluate

def _debug_wrapper(self, *args, **kwargs):
    print("[DEBUG] Running alignment check before evaluation ...")
    debug_positive_alignment(eval_dataset, tokenized_eval_dataset, tokenizer, n=2)
    return orig_evaluate(self, *args, **kwargs)
Trainer.evaluate = _debug_wrapper


trainer.train()
final_model_path = "./gemma2b_lora_toolformer_v4_final"
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)

print("Training Complete")
# === One-shot BOS / source_len debug (before dataset mapping) ===
try:
    bos_id = tokenizer.bos_token_id
    bos_tok = tokenizer.convert_ids_to_tokens([bos_id]) if bos_id is not None else [None]
    sample_raw = dataset[0]
    src_with = tokenizer(sample_raw["input"], truncation=True, max_length=128, add_special_tokens=True)
    src_no   = tokenizer(sample_raw["input"], truncation=True, max_length=128, add_special_tokens=False)
    print("[BOS-DEBUG] bos_token_id=", bos_id, "token=", bos_tok)
    print("[BOS-DEBUG] with_special_first_id=", src_with["input_ids"][0], "->", tokenizer.convert_ids_to_tokens([src_with["input_ids"][0]]))
    print("[BOS-DEBUG] len_with=", len(src_with["input_ids"]), "len_no=", len(src_no["input_ids"]))
    print("[BOS-DEBUG] starts_with_bos=", (bos_id is not None and src_with["input_ids"][0] == bos_id))
except Exception as e:
    print("[BOS-DEBUG] failed:", e)
