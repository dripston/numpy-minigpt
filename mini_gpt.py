import numpy as np

class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices):
        return ''.join([self.idx_to_char[i] for i in indices])


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def gelu_grad(x):
    c = np.sqrt(2 / np.pi)
    u = c * (x + 0.044715 * x ** 3)
    t = np.tanh(u)
    du = c * (1 + 3 * 0.044715 * x ** 2)
    return 0.5 * (1 + t) + 0.5 * x * (1 - t ** 2) * du


class Embedding:
    def __init__(self, vocab_size, embed_dim):
        self.token_embed = np.random.randn(vocab_size, embed_dim) * 0.02
        self.max_seq_len = 128
        self.position_embed = np.random.randn(self.max_seq_len, embed_dim) * 0.02
        self.grad_token_embed = np.zeros_like(self.token_embed)
        self.grad_position_embed = np.zeros_like(self.position_embed)

    def forward(self, token_ids):
        seq_len = len(token_ids)
        tok_emb = self.token_embed[token_ids]
        pos_emb = self.position_embed[:seq_len]
        self.last_token_ids = token_ids
        return tok_emb + pos_emb

    def backward(self, dout):
        self.grad_token_embed[:] = 0
        self.grad_position_embed[:] = 0
        for i, tid in enumerate(self.last_token_ids):
            self.grad_token_embed[tid] += dout[i]
        seq_len = dout.shape[0]
        self.grad_position_embed[:seq_len] += dout


class LayerNorm:
    def __init__(self, embed_dim):
        self.gamma = np.ones(embed_dim)
        self.beta = np.zeros(embed_dim)
        self.eps = 1e-5
        self.grad_gamma = np.zeros_like(self.gamma)
        self.grad_beta = np.zeros_like(self.beta)

    def forward(self, x):
        self.last_input = x
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.invstd = 1.0 / np.sqrt(self.var + self.eps)
        self.x_norm = (x - self.mean) * self.invstd
        return self.gamma * self.x_norm + self.beta

    def backward(self, dout):
        D = dout.shape[-1]
        self.grad_gamma = np.sum(dout * self.x_norm, axis=0)
        self.grad_beta = np.sum(dout, axis=0)
        dxnorm = dout * self.gamma
        dx = (1.0 / D) * self.invstd * (
            D * dxnorm
            - np.sum(dxnorm, axis=-1, keepdims=True)
            - self.x_norm * np.sum(dxnorm * self.x_norm, axis=-1, keepdims=True)
        )
        return dx


class SelfAttention:
    def __init__(self, embed_dim):
        scale = 0.02
        self.W_q = np.random.randn(embed_dim, embed_dim) * scale
        self.W_k = np.random.randn(embed_dim, embed_dim) * scale
        self.W_v = np.random.randn(embed_dim, embed_dim) * scale
        self.W_o = np.random.randn(embed_dim, embed_dim) * scale
        self.grad_W_q = np.zeros_like(self.W_q)
        self.grad_W_k = np.zeros_like(self.W_k)
        self.grad_W_v = np.zeros_like(self.W_v)
        self.grad_W_o = np.zeros_like(self.W_o)

    def forward(self, x):
        seq_len, embed_dim = x.shape
        self.last_input = x
        self.Q = np.dot(x, self.W_q)
        self.K = np.dot(x, self.W_k)
        self.V = np.dot(x, self.W_v)
        scores = np.dot(self.Q, self.K.T) / np.sqrt(embed_dim)
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * (-1e9)
        scores = scores + mask
        self.attn_weights = softmax(scores)
        self.attn_output = np.dot(self.attn_weights, self.V)
        return np.dot(self.attn_output, self.W_o)

    def backward(self, dout):
        x = self.last_input
        embed_dim = x.shape[1]
        self.grad_W_o = np.dot(self.attn_output.T, dout)
        d_attn_output = np.dot(dout, self.W_o.T)
        dV = np.dot(self.attn_weights.T, d_attn_output)
        d_attn_weights = np.dot(d_attn_output, self.V.T)
        dscores = self.attn_weights * (
            d_attn_weights - np.sum(d_attn_weights * self.attn_weights, axis=-1, keepdims=True)
        )
        dscores = dscores / np.sqrt(embed_dim)
        dQ = np.dot(dscores, self.K)
        dK = np.dot(dscores.T, self.Q)
        self.grad_W_q = np.dot(x.T, dQ)
        self.grad_W_k = np.dot(x.T, dK)
        self.grad_W_v = np.dot(x.T, dV)
        dx = np.dot(dQ, self.W_q.T) + np.dot(dK, self.W_k.T) + np.dot(dV, self.W_v.T)
        return dx


class FeedForward:
    def __init__(self, embed_dim):
        hidden_dim = embed_dim * 4
        scale = 0.02
        self.W1 = np.random.randn(embed_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, embed_dim) * scale
        self.b2 = np.zeros(embed_dim)
        self.grad_W1 = np.zeros_like(self.W1)
        self.grad_b1 = np.zeros_like(self.b1)
        self.grad_W2 = np.zeros_like(self.W2)
        self.grad_b2 = np.zeros_like(self.b2)

    def forward(self, x):
        self.last_input = x
        self.hidden = np.dot(x, self.W1) + self.b1
        self.hidden_activated = gelu(self.hidden)
        return np.dot(self.hidden_activated, self.W2) + self.b2

    def backward(self, dout):
        self.grad_W2 = np.dot(self.hidden_activated.T, dout)
        self.grad_b2 = np.sum(dout, axis=0)
        dh_act = np.dot(dout, self.W2.T)
        dhidden = dh_act * gelu_grad(self.hidden)
        self.grad_W1 = np.dot(self.last_input.T, dhidden)
        self.grad_b1 = np.sum(dhidden, axis=0)
        dx = np.dot(dhidden, self.W1.T)
        return dx


class TransformerBlock:
    def __init__(self, embed_dim):
        self.ln1 = LayerNorm(embed_dim)
        self.attention = SelfAttention(embed_dim)
        self.ln2 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim)

    def forward(self, x):
        normalized = self.ln1.forward(x)
        attended = self.attention.forward(normalized)
        x2 = x + attended
        normalized2 = self.ln2.forward(x2)
        fed_forward = self.ffn.forward(normalized2)
        x3 = x2 + fed_forward
        return x3

    def backward(self, dout):
        dff = dout
        dnormalized2 = self.ffn.backward(dff)
        dx2_b = self.ln2.backward(dnormalized2)
        dx2 = dout + dx2_b
        dattended = dx2
        dnormalized1 = self.attention.backward(dattended)
        dx_b = self.ln1.backward(dnormalized1)
        dx = dx2 + dx_b
        return dx

    def params_and_grads(self):
        return [
            (self.ln1.gamma, self.ln1.grad_gamma),
            (self.ln1.beta, self.ln1.grad_beta),
            (self.attention.W_q, self.attention.grad_W_q),
            (self.attention.W_k, self.attention.grad_W_k),
            (self.attention.W_v, self.attention.grad_W_v),
            (self.attention.W_o, self.attention.grad_W_o),
            (self.ln2.gamma, self.ln2.grad_gamma),
            (self.ln2.beta, self.ln2.grad_beta),
            (self.ffn.W1, self.ffn.grad_W1),
            (self.ffn.b1, self.ffn.grad_b1),
            (self.ffn.W2, self.ffn.grad_W2),
            (self.ffn.b2, self.ffn.grad_b2),
        ]


class MiniGPT:
    def __init__(self, vocab_size, embed_dim=64, num_blocks=2):
        self.embedding = Embedding(vocab_size, embed_dim)
        self.blocks = [TransformerBlock(embed_dim) for _ in range(num_blocks)]
        self.final_ln = LayerNorm(embed_dim)
        self.output_proj = np.random.randn(embed_dim, vocab_size) * 0.02
        self.grad_output_proj = np.zeros_like(self.output_proj)

    def forward(self, token_ids):
        x = self.embedding.forward(token_ids)
        for block in self.blocks:
            x = block.forward(x)
        x = self.final_ln.forward(x)
        self.last_final = x
        logits = np.dot(x, self.output_proj)
        return logits

    def backward(self, dlogits):
        self.grad_output_proj = np.dot(self.last_final.T, dlogits)
        dx = np.dot(dlogits, self.output_proj.T)
        dx = self.final_ln.backward(dx)
        for block in reversed(self.blocks):
            dx = block.backward(dx)
        self.embedding.backward(dx)

    def all_params_and_grads(self):
        params = [
            (self.output_proj, self.grad_output_proj),
            (self.final_ln.gamma, self.final_ln.grad_gamma),
            (self.final_ln.beta, self.final_ln.grad_beta),
            (self.embedding.token_embed, self.embedding.grad_token_embed),
            (self.embedding.position_embed, self.embedding.grad_position_embed),
        ]
        for block in self.blocks:
            params.extend(block.params_and_grads())
        return params

    def generate(self, tokenizer, prompt, max_new_tokens=200, temperature=0.8):
        token_ids = tokenizer.encode(prompt)
        for _ in range(max_new_tokens):
            context = token_ids[-self.embedding.max_seq_len:]
            logits = self.forward(np.array(context))
            last_logits = logits[-1] / temperature
            probs = softmax(last_logits)
            next_token = np.random.choice(len(probs), p=probs)
            token_ids.append(next_token)
        return tokenizer.decode(token_ids)


class Adam:
    def __init__(self, params_and_grads, lr=3e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p) for p, g in params_and_grads]
        self.v = [np.zeros_like(p) for p, g in params_and_grads]
        self.t = 0

    def step(self, params_and_grads):
        self.t += 1
        for i, (p, g) in enumerate(params_and_grads):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


if __name__ == "__main__":
    print("=" * 60)
    print("  MINI GPT — Full Backprop Through Every Layer")
    print("=" * 60)

    import os
    data_path = os.path.join(os.path.dirname(__file__), "tinyshakespeare.txt")

    if not os.path.exists(data_path):
        print(f"\n[ERROR] Dataset not found at: {data_path}")
        print("Download: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
        exit(1)

    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"\nDataset: {len(text):,} characters")

    tokenizer = CharTokenizer(text)
    print(f"Vocabulary: {tokenizer.vocab_size} unique characters")

    model = MiniGPT(vocab_size=tokenizer.vocab_size, embed_dim=64, num_blocks=2)
    optimizer = Adam(model.all_params_and_grads(), lr=3e-3)

    print("\n--- BEFORE TRAINING (random garbage) ---")
    print(model.generate(tokenizer, "ROMEO:", max_new_tokens=200))

    seq_length = 64
    num_steps = 3000
    data = np.array(tokenizer.encode(text))

    print("\n--- TRAINING ---")
    for step in range(num_steps):
        start = np.random.randint(0, len(data) - seq_length - 1)
        input_ids = data[start:start + seq_length]
        target_ids = data[start + 1:start + seq_length + 1]

        logits = model.forward(input_ids)
        probs = softmax(logits)
        correct_probs = probs[np.arange(seq_length), target_ids]
        loss = -np.mean(np.log(correct_probs + 1e-8))

        dlogits = probs.copy()
        dlogits[np.arange(seq_length), target_ids] -= 1
        dlogits /= seq_length

        model.backward(dlogits)
        optimizer.step(model.all_params_and_grads())

        if step % 100 == 0:
            print(f"  Step {step:4d} | Loss: {loss:.4f}")

    print("\n--- AFTER TRAINING ---")
    print(model.generate(tokenizer, "ROMEO:", max_new_tokens=300, temperature=0.8))

    save_path = os.path.join(os.path.dirname(__file__), "model_weights.npz")
    weights = {}
    weights["output_proj"] = model.output_proj
    weights["final_ln_gamma"] = model.final_ln.gamma
    weights["final_ln_beta"] = model.final_ln.beta
    weights["token_embed"] = model.embedding.token_embed
    weights["position_embed"] = model.embedding.position_embed
    for i, block in enumerate(model.blocks):
        weights[f"block{i}_ln1_gamma"] = block.ln1.gamma
        weights[f"block{i}_ln1_beta"] = block.ln1.beta
        weights[f"block{i}_Wq"] = block.attention.W_q
        weights[f"block{i}_Wk"] = block.attention.W_k
        weights[f"block{i}_Wv"] = block.attention.W_v
        weights[f"block{i}_Wo"] = block.attention.W_o
        weights[f"block{i}_ln2_gamma"] = block.ln2.gamma
        weights[f"block{i}_ln2_beta"] = block.ln2.beta
        weights[f"block{i}_W1"] = block.ffn.W1
        weights[f"block{i}_b1"] = block.ffn.b1
        weights[f"block{i}_W2"] = block.ffn.W2
        weights[f"block{i}_b2"] = block.ffn.b2

    np.savez(save_path, num_blocks=len(model.blocks), **weights)

    tokenizer_path = os.path.join(os.path.dirname(__file__), "tokenizer.npz")
    np.savez(tokenizer_path, chars=np.array(tokenizer.chars))

    print(f"\nSaved weights to {save_path}")
    print(f"Saved tokenizer to {tokenizer_path}")