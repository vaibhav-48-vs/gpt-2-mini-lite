import math
import time

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

######
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# ---------------- precision setup ----------------
# TF32 for any matmuls still running in fp32 (Ampere+)
torch.set_float32_matmul_precision("high")

use_amp = device.type == "cuda"
# bf16 needs Ampere+ (sm80: A100/3090/4090/H100). Older GPUs (T4/V100) -> fp16 + GradScaler
if use_amp and torch.cuda.get_device_capability()[0] >= 8:
    amp_dtype = torch.bfloat16
else:
    amp_dtype = torch.float16
print(f"autocast: {use_amp} | dtype: {amp_dtype if use_amp else torch.float32}")


def amp_ctx():
    return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp)


# ---------------- model ----------------
class RoPE(nn.Module):
    """Precomputed cos/sin tables; applied to (B, H, T, head_dim)."""

    def __init__(self, head_dim, max_len, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))  # ω_i
        angles = torch.outer(torch.arange(max_len).float(), inv_freq)                # θ = m * ω_i
        self.register_buffer("cos", angles.cos(), persistent=False)                  # (T, hd/2)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x):
        T = x.shape[-2]
        cos, sin = self.cos[:T], self.sin[:T]
        # rotate in fp32 for accuracy, cast back to the autocast dtype
        xf = x.float().unflatten(-1, (-1, 2))       # pairs (x0,x1), (x2,x3), ...
        x1, x2 = xf[..., 0], xf[..., 1]
        out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return out.flatten(-2).type_as(x)


class MQA(nn.Module):
    """Multi-Query Attention: H query heads, one shared K/V head. All heads in one batched op."""

    def __init__(self, embed, heads, block_size):
        super().__init__()
        assert embed % heads == 0
        self.heads = heads
        self.hsize = embed // heads
        self.q = nn.Linear(embed, embed)              # all H query heads at once
        self.kv = nn.Linear(embed, 2 * self.hsize)    # single shared K and V
        self.proj = nn.Linear(embed, embed)
        self.rope = RoPE(self.hsize, block_size)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.heads, self.hsize).transpose(1, 2)  # (B, H, T, hs)
        k, v = self.kv(x).split(self.hsize, dim=-1)                        # (B, T, hs)
        k, v = k.unsqueeze(1), v.unsqueeze(1)                              # (B, 1, T, hs)

        q = self.rope(q)
        k = self.rope(k)                                                   # rotated once, not per head

        # broadcast shared K/V to all heads (expand = view, no copy)
        k = k.expand(B, self.heads, T, self.hsize)
        v = v.expand(B, self.heads, T, self.hsize)

        # fused causal attention (Flash / mem-efficient kernel), scaling by 1/sqrt(hs) built in
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, heads, embed, block_size):
        super().__init__()
        self.layerNorm = nn.LayerNorm(embed)       # fused kernel
        self.layerNorm2 = nn.LayerNorm(embed)
        self.maskedAttention = MQA(embed, heads, block_size)
        self.ff = nn.Sequential(
            nn.Linear(embed, 4 * embed),
            nn.GELU(),
            nn.Linear(4 * embed, embed),
        )

    def forward(self, x):
        x = x + self.maskedAttention(self.layerNorm(x))
        x = x + self.ff(self.layerNorm2(x))
        return x

class Decoder(nn.Module):
    def __init__(self, n_blocks, heads, embed, vocab_size, block_size):
        super().__init__()
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, embed)
        self.blocks = nn.ModuleList(
            [Block(heads, embed, block_size) for _ in range(n_blocks)]
        )
        self.layerNorm = nn.LayerNorm(embed)
        self.ff = nn.Linear(embed, vocab_size, bias=False)

        # Weight tying
        self.ff.weight = self.embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x):
        assert x.shape[1] <= self.block_size, "sequence longer than block_size"
        x1 = self.embedding(x)
        for Tblock in self.blocks:
            x1 = Tblock(x1)
        x1 = self.layerNorm(x1)
        return self.ff(x1)


# ---------------- data ----------------
def load_tokens(path="shakespeare.txt"):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    enc = tiktoken.get_encoding("gpt2")
    return torch.tensor(enc.encode(text), dtype=torch.long)

import numpy as np

class dataloaderlite:
    def __init__(self, tokens, b, t):
        self.b = b
        self.t = t
        self.tokens = tokens
        self.l = len(tokens)
        assert self.l > b * t + 1, "not enough tokens for one batch"
        self.currpos = 0
        self.stokens = np.random.permutation(np.arange(0, self.l - (b*t+1), b*t+1))


    def reset(self):
        self.currpos = 0

    def getbatch(self):
        b, t = self.b, self.t
        idx=self.stokens[self.currpos]
        d = self.tokens[idx : idx + b * t + 1]
        x = d[:-1].view(b, t)
        y = d[1:].view(b, t)
        self.currpos += 1 # FIX: was b*t + 1 (skipped a token every batch)
        if self.currpos >=len(self.stokens):
            self.stokens = np.random.permutation(np.arange(0, self.l - (b*t+1), b*t+1))
            self.currpos = 0
        return x, y


# ---------------- hyperparameters ----------------
vocab_size = 50304      # 50257 padded up to a multiple of 128 -> faster matmuls
batch_size = 64
block_size = 128

max_steps = 6000
warmup_iters = 500
max_lr = 3e-4
min_lr = 3e-5
weight_decay = 0.01
grad_clip = 1.0

eval_interval = 250
eval_iters = 20
log_interval = 10


def lr_lambda(step):
    if step < warmup_iters:
        return (step + 1) / warmup_iters
    progress = min((step - warmup_iters) / max(1, max_steps - warmup_iters), 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return (min_lr / max_lr) + (1 - min_lr / max_lr) * cosine


# ---------------- setup ----------------
tokens = load_tokens("/kaggle/input/datasets/kaggle4848/datasets-acosharma-literature/main.txt").to(device)   # whole dataset lives on GPU
n_train = int(0.9 * len(tokens))
train_data = dataloaderlite(tokens, batch_size, block_size)
# val_data = dataloaderlite(tokens[n_train:], batch_size, block_size)
# print(f"train tokens: {train_data.l:,} | val tokens: {val_data.l:,}")

raw_model = Decoder(
    n_blocks=6, heads=4, embed=256, vocab_size=vocab_size, block_size=block_size
).to(device)                                          # params stay fp32 (master weights)
print(f"params: {sum(p.numel() for p in raw_model.parameters()) / 1e6:.2f}M")

decay_params = [p for p in raw_model.parameters() if p.requires_grad and p.dim() >= 2]
no_decay_params = [p for p in raw_model.parameters() if p.requires_grad and p.dim() < 2]
optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=max_lr,
    betas=(0.9, 0.95),
    fused=(device.type == "cuda"),                    # single fused kernel for the update
)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Only needed for fp16 (bf16 has fp32's exponent range -> no loss scaling)
scaler = torch.amp.GradScaler(device.type, enabled=(use_amp and amp_dtype == torch.float16))

# torch.compile: fuses elementwise ops / kernels. First step is slow (compilation).
model = torch.compile(raw_model) if device.type == "cuda" else raw_model


@torch.no_grad()
def estimate_val_loss():
    model.eval()
    val_data.reset()
    n = min(eval_iters, (val_data.l - 1) // (batch_size * block_size))
    total = 0.0
    for _ in range(n):
        x, y = val_data.getbatch()
        with amp_ctx():
            logits = model(x)
            total += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()
    model.train()
    return total / n


def sync():
    if device.type == "cuda":
        torch.cuda.synchronize()


# ---------------- training ----------------
model.train()
for step in range(max_steps):
    t0 = time.time()

    x, y = train_data.getbatch()
    optimizer.zero_grad(set_to_none=True)

    with amp_ctx():
        logits = model(x)                                  # bf16/fp16 activations
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))  # autocast runs this in fp32

    scaler.scale(loss).backward()                          # no-op scaling when bf16
    scaler.unscale_(optimizer)                             # so clipping sees true grads
    norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)

    lr = scheduler.get_last_lr()[0]
    scaler.step(optimizer)                                 # == optimizer.step() when bf16
    scaler.update()
    scheduler.step()

    sync()
    dt = (time.time() - t0) * 1000

    if step % log_interval == 0:
        tokens_per_sec = (train_data.b * train_data.t * 1000) / dt
        print(
            f"| step {step} | train loss = {loss.item():.4f} | norm = {norm.item():.3f} "
            f"| time : {dt:.2f}ms | lr = {lr:.6f} | tok/sec: {tokens_per_sec:.2f}"
        )

    # if step % eval_interval == 0 or step == max_steps - 1:
    #     print(f"| step {step} | val loss = {estimate_val_loss():.4f}")

# Save from raw_model (compiled wrapper prefixes keys with "_orig_mod.")
torch.save(raw_model.state_dict(), "ckpt.pt")