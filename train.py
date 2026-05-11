import argparse
import os
import time

import torch
from torch import nn

from prepare import (
    BATCH_SIZE,
    BLOCK_SIZE,
    DEVICE,
    LR,
    MAX_TRAIN_SECONDS,
    VOCAB_SIZE,
    evaluate_bpb,
    get_batch,
    train_ids,
)

EXPERIMENT_LR = 1e-3


class TinyGPT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.n_embd = 128
        self.n_head = 4
        self.n_layer = 2

        self.tok_emb = nn.Embedding(VOCAB_SIZE, self.n_embd)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, self.n_embd)
        self.drop = nn.Dropout(0.05)
        self.blocks = nn.ModuleList([self._build_block() for _ in range(self.n_layer)])
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.head = nn.Linear(self.n_embd, VOCAB_SIZE)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(BLOCK_SIZE, BLOCK_SIZE, dtype=torch.bool), diagonal=1),
        )

    def _build_block(self) -> nn.Module:
        return nn.ModuleDict(
            {
                "ln1": nn.LayerNorm(self.n_embd),
                "attn": nn.MultiheadAttention(self.n_embd, self.n_head, dropout=0.05, batch_first=True),
                "ln2": nn.LayerNorm(self.n_embd),
                "mlp": nn.Sequential(
                    nn.Linear(self.n_embd, 4 * self.n_embd),
                    nn.GELU(),
                    nn.Linear(4 * self.n_embd, self.n_embd),
                    nn.Dropout(0.05),
                ),
            }
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = idx.shape
        positions = torch.arange(seq_len, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(positions)
        x = self.drop(x)
        for block in self.blocks:
            x = x + block["attn"](
                block["ln1"](x),
                block["ln1"](x),
                block["ln1"](x),
                attn_mask=self.mask[:seq_len, :seq_len],
                need_weights=False,
            )[0]
            x = x + block["mlp"](block["ln2"](x))
        x = self.ln_f(x)
        return self.head(x)


def generate_sample(model: nn.Module, max_new_tokens: int = 192) -> str:
    model.eval()
    with torch.no_grad():
        idx = torch.randint(0, VOCAB_SIZE, (1, 1), device=DEVICE)
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= BLOCK_SIZE else idx[:, -BLOCK_SIZE:]
            logits = model(idx_cond)
            probs = torch.softmax(logits[:, -1, :] / 1.0, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
    return bytes(idx[0].tolist()).decode("utf-8", errors="ignore")


def write_samples(samples: list[str]) -> None:
    result = "\n\n---\n\n".join(samples)
    print(result)


def main(sample_mode: bool = False) -> None:
    model = TinyGPT().to(DEVICE)
    if sample_mode:
        samples = [generate_sample(model, max_new_tokens=128) for _ in range(5)]
        write_samples(samples)
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=EXPERIMENT_LR, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    start_time = time.time()
    step = 0
    model.train()
    while time.time() - start_time < MAX_TRAIN_SECONDS:
        xb, yb = get_batch("train")
        logits = model(xb)
        loss = loss_fn(logits.view(-1, VOCAB_SIZE), yb.view(-1))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        step += 1

    training_seconds = time.time() - start_time
    val_bpb = evaluate_bpb(model)
    total_seconds = time.time() - start_time

    if DEVICE.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(DEVICE) / 1024 ** 2
    else:
        peak_memory_mb = 0.0

    num_params_M = sum(p.numel() for p in model.parameters()) / 1e6

    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"training_seconds: {training_seconds:.2f}")
    print(f"total_seconds:    {total_seconds:.2f}")
    print(f"peak_memory_mb:   {peak_memory_mb:.2f}")
    print(f"num_params_M:     {num_params_M:.3f}")
    print(f"num_steps:        {step}")

    os.makedirs("samples", exist_ok=True)
    sample_text = generate_sample(model, max_new_tokens=192)
    with open(os.path.join("samples", "samples_final.txt"), "w", encoding="utf-8") as out_file:
        out_file.write(sample_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Generate sample text instead of training.")
    args = parser.parse_args()
    main(sample_mode=args.sample)
