import re
import random
import json
from datasets import load_dataset

OUTPUT_POSITIVE_JSONL = "exp2_positive_samples.jsonl"
OUTPUT_NEGATIVE_JSONL = "exp2_negative_samples.jsonl"
OUTPUT_COMBINED_JSONL = "exp2_combined_dataset_ood.jsonl"
OUTPUT_OOD_JSONL = "exp2_ood_tool_samples.jsonl"
NUM_POSITIVE = 20
NUM_NEGATIVE = 20
NUM_OOD = 20
C4_SPLIT = "train"  # or "validation"
WITH_CONTEXT = True

POSITIVE_LABEL = "positive"
NEGATIVE_LABEL = "negative"
TOOL_OOD_LABEL = "tool_ood"
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
OOD_TOOL_TEMPLATES = [

    "Without doing the arithmetic yourself, issue a single tool call that asks the calculator to evaluate {expr}.",
    "Formulate exactly one <tool_call> that makes the calculator compute {expr}, and output only that tool call.",
    "Use the calculator tool (not mental math) to obtain the value of {expr}. Respond with a <tool_call>.",
    "Act as an orchestrator: delegate the computation of {expr} to the calculator tool and return only the tool invocation.",
    "Prepare a tool call for the calculator so that it can evaluate {expr}. Do not simplify the expression by hand.",

    "I'm wiring this into a larger system that expects a tool call. Please wrap the expression {expr} in a <tool_call> for the calculator.",
    "Imagine you are building a pipeline: your job is to generate the <tool_call> that tells the calculator to compute {expr}.",
    "For logging purposes, we just need the tool call, not the numeric result. Produce a single <tool_call> for {expr}.",

    "Instead of computing the answer, construct a calculator invocation that will evaluate {expr} when executed.",
    "Translate the math query {expr} into a calculator tool call, enclosed in <tool_call>...</tool_call>."
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

        samples.append({
            "input": full_input,
            "target": target,
            "label": POSITIVE_LABEL
        })
    return samples

def generate_ood_tool_samples(num_samples=100):
    """
    Generate out-of-distribution (OOD) tool call samples:

    """
    samples = []
    ops = ["+", "-", "*", "/"]
    while len(samples) < num_samples:

        a = random.randint(1, 999)
        b = random.randint(2, 999)  
        op = random.choice(ops)
        if op == "/" and b == 0:
            continue
        expr = f"{a}{op}{b}"

        prompt_template = random.choice(OOD_TOOL_TEMPLATES)
        full_input = prompt_template.format(expr=expr)

        target = f"<tool_call>calculator({expr})</tool_call>"

        samples.append({
            "input": full_input,
            "target": target,
            "label": TOOL_OOD_LABEL
        })
    return samples


# negative samples
def is_negative_sample(text, min_len=30, max_len=150):
    text_stripped = text.strip()
    return (
        bool(re.search(r"\d", text)) and              # contain numbers
        not re.search(r"[+\-*/=]", text) and          # not contain operators
        # not re.search(r"\b(sum|plus|subtract|minus||multiply|times|divide)\b", text, re.IGNORECASE) and  # not contain math keywords
        not text_stripped[0].isdigit() and               # not start with a number
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
                neg_samples.append({
                    "input": line,
                    "target": line,
                    "label": NEGATIVE_LABEL
                })
                if len(neg_samples) >= num_samples:
                    return neg_samples
    return neg_samples


if __name__ == "__main__":
    pos_samples = generate_positive_samples(NUM_POSITIVE, with_context=WITH_CONTEXT)
    with open(OUTPUT_POSITIVE_JSONL, "w") as f:
        for item in pos_samples:
            f.write(json.dumps(item) + "\n")
    print(f"Positive samples saved to：{OUTPUT_POSITIVE_JSONL}")

    neg_samples = sample_negative_from_c4(NUM_NEGATIVE, c4_split=C4_SPLIT)
    with open(OUTPUT_NEGATIVE_JSONL, "w") as f:
        for item in neg_samples:
            f.write(json.dumps(item) + "\n")
    print(f"Negative samples saved to：{OUTPUT_NEGATIVE_JSONL}")

    ood_samples = generate_ood_tool_samples(NUM_OOD)
    with open(OUTPUT_OOD_JSONL, "w") as f:
        for item in ood_samples:
            f.write(json.dumps(item) + "\n")
    print(f"OOD tool samples saved to：{OUTPUT_OOD_JSONL}")

    combined = pos_samples + neg_samples + ood_samples
    random.shuffle(combined)
    with open(OUTPUT_COMBINED_JSONL, "w") as f:
        for item in combined:
            f.write(json.dumps(item) + "\n")
    print(f"Combined samples saved to：{OUTPUT_COMBINED_JSONL}")
