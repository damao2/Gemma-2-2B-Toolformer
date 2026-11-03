import random
import re

"""
Generate a corpus focused on tool-call contexts for activation collection.

This script is separate from `gen_top_activations_corpus.py` and is tailored to:
  1) Produce positive examples in the same format as training:
       prompt + "\n" + "<tool_call>calculator(a op b)</tool_call>"
  2) Produce hard negatives where the `<tool_call>...` string appears in the
     text as a quoted / described string, NOT as an actual tool invocation.

You can then run `collect_top_activations.py` with `TEXT_PATH` pointing to the
output file from this script, and merge the resulting Neuron/Activation rows
with your existing database.
"""

# Number of samples to generate
NUM_POSITIVE = 300        # tool-call intent: prompt + "\n" + target
NUM_HARD_NEGATIVE = 100   # contains `<tool_call>...` but not as a real call
NUM_SOFT_NEGATIVE = 100   # mentions calculator/tool, but no `<tool_call>` at all

OUTPUT_FILE = "toolcall_activations_corpus_1k.txt"

WITH_CONTEXT = True


# Reuse / adapt the same templates from gen_top_activations_corpus.py
question_templates = {
    "+": [
        "What is {a} + {b}?",
        "Could you calculate {a} plus {b} for me?",
        "What's the total of {a} and {b}?",
        "If you add {a} and {b}, what do you get?",
        "Tell me the result of {a} + {b}.",
        "How much is {a} plus {b}?",
        "Compute the value of {a} + {b}.",
        "If I have {a} items and I get {b} more, I need you to calculate {a} + {b}.",
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
    ],
}

question_templates_2 = {
    "prompts": [
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
    "I need help with this calculation.",
]


def _sample_arith_prompt(with_context: bool = True) -> tuple[str, str, str, int, int]:
    """Sample (full_input, op, expr, a, b) using the same style as training.

    full_input is what you used as `prompt` during training, without any
    `User:` / `Assistant:` prefixes.
    """

    ops = list(question_templates.keys())
    a = random.randint(1, 999)
    b = random.randint(1, 999)
    op = random.choice(ops)

    # avoid division by zero
    if op == "/" and b == 0:
        b = random.randint(1, 999) or 1

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
    return full_input, op, expr, a, b


def generate_toolcall_positive_samples(num_samples: int, with_context: bool = True) -> list[str]:
    """Generate positive samples in `prompt + \n + target` format.

    Each line looks like:
      "Tell me the result of 603 + 560.\n<tool_call>calculator(603+560)</tool_call>"
    """

    samples: list[str] = []
    while len(samples) < num_samples:
        full_input, op, expr, a, b = _sample_arith_prompt(with_context=with_context)
        # Reuse the exact target format you trained on, but encode the
        # separator as a literal "\\n" so the whole sample stays on one line.
        target = f"<tool_call>calculator({expr})</tool_call>"
        # Make sure there are no real newlines inside the prompt portion.
        full_input = full_input.replace("\n", " ")
        full_text = full_input + r"\n" + target
        samples.append(full_text)
    return samples


# Hard negatives: `<tool_call>...` appears as a string in the description,
# but the user is *not* actually asking the model to call the tool.
hard_negative_templates = [
    "I saw the string <tool_call>calculator({expr})</tool_call> in some logs. What does it mean, in plain English?",
    "Please *do not* actually run <tool_call>calculator({expr})</tool_call>; just explain what this notation represents.",
    "In the documentation, there's an example like <tool_call>calculator({expr})</tool_call>. Describe what this example is doing.",
    "Quote the following exactly, without executing it: <tool_call>calculator({expr})</tool_call>.",
    "Consider the literal text '<tool_call>calculator({expr})</tool_call>'. Explain when a system might output this, but don't perform the calculation.",
]


def generate_toolcall_hard_negatives(num_samples: int, with_context: bool = True) -> list[str]:
    """Generate negatives that mention `<tool_call>...` as text, not as a command.

    These are single-line prompts, e.g.:
      "I saw the string <tool_call>calculator(3+4)</tool_call> in some logs. What does it mean?"
    """

    samples: list[str] = []
    while len(samples) < num_samples:
        full_input, op, expr, a, b = _sample_arith_prompt(with_context=with_context)
        # We only need expr here; full_input is unused but keeps distribution similar
        template = random.choice(hard_negative_templates)
        text = template.format(expr=expr)
        samples.append(text)
    return samples


# Soft negatives: mention calculator/tool in a generic way, but never use
# `<tool_call>` markup. These are intended to help separate "has a calculator
# concept" from "is actually in tool_call format".
soft_negative_templates = [
    "You can use a calculator tool to solve expressions like {expr}, but for now just explain the reasoning in words.",
    "Describe how a calculator tool might evaluate an expression such as {expr}, without giving the final numeric answer.",
    "In this system, the assistant has access to tools such as a calculator. Explain what the calculator tool is for.",
    "Write a short paragraph about when it is appropriate to use a calculator tool instead of doing mental math.",
    "List some situations where a calculator is useful, and some where using a tool would be unnecessary.",
    "Explain to a student how they could double-check their work on {expr} using a calculator, but do not actually compute it.",
]


def generate_toolcall_soft_negatives(num_samples: int, with_context: bool = True) -> list[str]:
    """Generate negatives that mention calculator/tools but have no `<tool_call>`.

    These are also single-line prompts, e.g.:
      "You can use a calculator tool to solve expressions like 3+4, but ..."
    """

    samples: list[str] = []
    while len(samples) < num_samples:
        full_input, op, expr, a, b = _sample_arith_prompt(with_context=with_context)
        template = random.choice(soft_negative_templates)
        text = template.format(expr=expr)
        samples.append(text)
    return samples


def write_to_txt(filename: str, data_list: list[str]) -> None:
    """Write a list of texts to a txt file, one sample per line."""

    with open(filename, "w", encoding="utf-8") as f:
        for item in data_list:
            # Trim outer whitespace but keep internal content as-is.
            item = item.strip()
            if not item:
                continue
            # Positive samples already contain a literal "\\n" between
            # prompt and <tool_call>..., and we've removed any real
            # newlines from them. Negatives never had newlines.
            # So we can safely write one sample per physical line.
            f.write(item + "\n")


if __name__ == "__main__":
    print("--- Generating tool-call focused activation corpus ---")
    print(f"  Positive (prompt + \\n + target): {NUM_POSITIVE}")
    print(f"  Hard negatives (quoted <tool_call>...): {NUM_HARD_NEGATIVE}")
    print(f"  Soft negatives (calculator/tool, no <tool_call>): {NUM_SOFT_NEGATIVE}")
    print(f"  Output file: {OUTPUT_FILE}")
    print("-" * 60)

    pos_samples = generate_toolcall_positive_samples(NUM_POSITIVE, with_context=WITH_CONTEXT)
    hard_neg_samples = generate_toolcall_hard_negatives(NUM_HARD_NEGATIVE, with_context=WITH_CONTEXT)
    soft_neg_samples = generate_toolcall_soft_negatives(NUM_SOFT_NEGATIVE, with_context=WITH_CONTEXT)

    combined: list[str] = pos_samples + hard_neg_samples + soft_neg_samples
    random.shuffle(combined)

    print(f"Writing {len(combined)} samples to {OUTPUT_FILE}...")
    write_to_txt(OUTPUT_FILE, combined)

    print("\nTool-call activation corpus generation complete!")
