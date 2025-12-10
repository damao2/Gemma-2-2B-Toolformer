import re
import random
import json
from datasets import load_dataset

NUM_POSITIVE = 30  # The number of "positive" (tool-call intent) samples
NUM_NEGATIVE = 30   # The number of "negative" (general natural language) samples

# 2. Dataset split ratio
VALIDATION_SPLIT = 0.1  # The proportion of data used for validation (e.g., 0.1 means 10%)

# 3. Output file names
OUTPUT_TRAIN_FILE = "sae_train.txt"
OUTPUT_VALIDATION_FILE = "sae_validation.txt"

# 4. Other advanced settings
C4_SPLIT = "train"  # Which part of the C4 dataset to sample from ('train' or 'validation')
WITH_CONTEXT = True # Whether to include context sentences in positive samples


# positive samples
question_templates = {
    "+": [
        "What is {a} + {b}?",
        "Could you calculate {a} plus {b} for me?",
        "What's the total of {a} and {b}?",
        "If you add {a} and {b}, what do you get?",
        "Tell me the result of {a} + {b}.",
        "How much is {a} plus {b}?",
        "Compute the value of {a} + {b}.",
        "If I have {a} items and I get {b} more, I need you to calculate {a} + {b}."
    ],
    "-": [
        "What is {a} - {b}?",
        "Could you subtract {b} from {a}?",
        "What's the difference between {a} and {b}?",
        "If you take {b} away from {a}, what do you get?",
        "Tell me the result of {a} minus {b}.",
        "Compute the value of {a} - {b}.",
        "I started with a budget of {a} and spent {b}. To see what's left, calculate {a} - {b}.",

    ],
    "*": [
        "What is {a} times {b}?",
        "Could you multiply {a} and {b} for me?",
        "What's the product of {a} and {b}?",
        "If you multiply {a} by {b}, what do you get?",
        "Compute the value of {a} * {b}.",
        "To find the area of the rectangle, I need to multiply its sides. Calculate {a} * {b} for me.",

    ],
    "/": [
        "What is {a} divided by {b}?",
        "Could you divide {a} by {b} for me?",
        "If you divide {a} by {b}, what do you get?",
        "Please compute the result of {a} over {b}.",
        "Compute the value of {a} / {b}.",
        "The bill was split {b} ways. The total was {a}. Please compute {a} / {b} to see what each person owes.",
    ]
}
question_templates_2 = {
"prompts":[
"Calculate {X} {op} {Y}.",
"Give me the result of {X} {op} {Y}.",
"Solve for {X} {op} {Y}.",
"Run the numbers for {X} {op} {Y}.",
"Tell me the answer to {X} {op} {Y}.",
"Do the math for {X} {op} {Y}.",
"I need to know the answer to {X} {op} {Y}.",
"I'm trying to figure out {X} {op} {Y}.",
"I'm stuck on this calculation: {X} {op} {Y}.",
"Let's find out what {X} {op} {Y} equals.",
"I'd like to know the result of {X} {op} {Y}.",
"My homework asks for the value of {X} {op} {Y}.",
]
}
contexts = [
    "I'm helping my brother learn math.",
    "While budgeting my expenses, I got stuck.",
    "This is a quick math exercise.",
    "Here's a math problem for you.",
    "I need help with this calculation."
]

def generate_positive_samples(num_samples=100, with_context=True):
    samples = []
    ops = list(question_templates.keys())
    while len(samples) < num_samples:
        a = random.randint(1, 999)
        b = random.randint(1, 999)
        op = random.choice(ops)

        if op == "/" and b == 0:
            continue
        if random.random() > 0.5:
            question = random.choice(question_templates[op]).format(a=a, b=b)
        else:
            question = random.choice(question_templates_2["prompts"]).format(X=a, Y=b, op=op)

        context = random.choice(contexts) + " " if with_context else ""

        if random.random() > 0.5:
            full_input = context + question
        else:
            full_input = question    
        expr = f"{a}{op}{b}"
        target = f"<tool_call>calculator({expr})</tool_call>"

        samples.append({"input": full_input}) # only need 'input' field
    return samples

# negative samples
def is_negative_sample(text, min_len=30, max_len=150):
    return (
        bool(re.search(r"\d", text)) and              # contain numbers
        not re.search(r"[+\-*/=]", text) and          # not contain operators
        min_len <= len(text.strip()) <= max_len       # within length req
    )

def sample_negative_from_c4(num_samples=100, c4_split='train'):
    print("Finding in c4..")
    c4 = load_dataset("allenai/c4", "en", split=c4_split, streaming=True)
    neg_samples = []
    for example in c4:
        text = example["text"]
        for line in text.split('\n'):
            line = line.strip()
            if is_negative_sample(line):
                neg_samples.append({"input": line}) # only need 'input' field
                if len(neg_samples) >= num_samples:
                    return neg_samples
    return neg_samples

def write_to_txt(filename, data_list):
    """Write a list of input texts to a txt file, one sample per line"""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(item.strip() + '\n')

if __name__ == "__main__":
    print("--- Starting Dataset Generation with the following settings: ---")
    print(f"  Positive Samples: {NUM_POSITIVE}")
    print(f"  Negative Samples: {NUM_NEGATIVE}")
    print(f"  Validation Split: {VALIDATION_SPLIT:.0%}")
    print(f"  Train File:       {OUTPUT_TRAIN_FILE}")
    print(f"  Validation File:  {OUTPUT_VALIDATION_FILE}")
    print("-" * 60)

    pos_samples = generate_positive_samples(NUM_POSITIVE, with_context=WITH_CONTEXT)
    neg_samples = sample_negative_from_c4(NUM_NEGATIVE, c4_split=C4_SPLIT)
    
    # Extract the values corresponding to the 'input' key from the list of dictionaries to get a list of strings
    combined = [item['input'] for item in pos_samples] + [item['input'] for item in neg_samples]
    random.shuffle(combined)
    # 3. Split into training and validation sets
    split_index = int(len(combined) * (1 - VALIDATION_SPLIT))
    train_data = combined[:split_index]
    validation_data = combined[split_index:]

    # 4. Write to final txt files
    print(f"Writing {len(train_data)} samples to {OUTPUT_TRAIN_FILE}...")
    write_to_txt(OUTPUT_TRAIN_FILE, train_data)

    print(f"Writing {len(validation_data)} samples to {OUTPUT_VALIDATION_FILE}...")
    write_to_txt(OUTPUT_VALIDATION_FILE, validation_data)
    
    print("\nDataset generation complete!")
