import os
import numpy as np

from mini_gpt import (
    CharTokenizer,
    MiniGPT,
    softmax,
)


def load_model(weights_path, tokenizer_path):
    data = np.load(weights_path)
    tok_data = np.load(tokenizer_path, allow_pickle=True)

    chars = list(tok_data["chars"])
    vocab_size = len(chars)
    embed_dim = data["token_embed"].shape[1]
    num_blocks = int(data["num_blocks"])

    tokenizer = CharTokenizer("")
    tokenizer.chars = chars
    tokenizer.vocab_size = vocab_size
    tokenizer.char_to_idx = {ch: i for i, ch in enumerate(chars)}
    tokenizer.idx_to_char = {i: ch for i, ch in enumerate(chars)}

    model = MiniGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_blocks=num_blocks,
    )

    model.output_proj = data["output_proj"]
    model.final_ln.gamma = data["final_ln_gamma"]
    model.final_ln.beta = data["final_ln_beta"]

    model.embedding.token_embed = data["token_embed"]
    model.embedding.position_embed = data["position_embed"]

    for i, block in enumerate(model.blocks):
        block.ln1.gamma = data[f"block{i}_ln1_gamma"]
        block.ln1.beta = data[f"block{i}_ln1_beta"]

        block.attention.W_q = data[f"block{i}_Wq"]
        block.attention.W_k = data[f"block{i}_Wk"]
        block.attention.W_v = data[f"block{i}_Wv"]
        block.attention.W_o = data[f"block{i}_Wo"]

        block.ln2.gamma = data[f"block{i}_ln2_gamma"]
        block.ln2.beta = data[f"block{i}_ln2_beta"]

        block.ffn.W1 = data[f"block{i}_W1"]
        block.ffn.b1 = data[f"block{i}_b1"]
        block.ffn.W2 = data[f"block{i}_W2"]
        block.ffn.b2 = data[f"block{i}_b2"]

    return model, tokenizer


def generate_stream(
    model,
    tokenizer,
    prompt="ROMEO:",
    max_new_tokens=3000,
    temperature=0.8,
):
    token_ids = tokenizer.encode(prompt)

    generated = prompt

    print(prompt, end="", flush=True)

    for _ in range(max_new_tokens):
        context = token_ids[-model.embedding.max_seq_len:]

        logits = model.forward(np.array(context))

        last_logits = logits[-1] / temperature

        probs = softmax(last_logits)

        next_token = np.random.choice(len(probs), p=probs)

        token_ids.append(next_token)

        ch = tokenizer.idx_to_char[next_token]

        generated += ch

        print(ch, end="", flush=True)

    print()

    return generated


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)

    weights_path = os.path.join(base_dir, "model_weights.npz")
    tokenizer_path = os.path.join(base_dir, "tokenizer.npz")

    if not os.path.exists(weights_path):
        print("[ERROR] model_weights.npz not found.")
        print("Run mini_gpt.py first.")
        exit(1)

    if not os.path.exists(tokenizer_path):
        print("[ERROR] tokenizer.npz not found.")
        print("Run mini_gpt.py first.")
        exit(1)

    model, tokenizer = load_model(weights_path, tokenizer_path)

    print("=" * 70)
    print("          MINI GPT — Shakespeare Generation")
    print("=" * 70)
    print()

    generated_text = generate_stream(
        model=model,
        tokenizer=tokenizer,
        prompt="ROMEO:",
        max_new_tokens=3000,
        temperature=0.8,
    )

    output_path = os.path.join(base_dir, "generated_shakespeare.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(generated_text)

    print("\n" + "=" * 70)
    print(f"Generation complete!")
    print(f"Saved to: {output_path}")
    print("=" * 70)