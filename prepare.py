import os
import random
import time

import numpy as np
import torch
from datasets import load_dataset

DATASET_NAME = "karpathy/tinystories-gpt4-clean"
TRAIN_SIZE = 2000
VAL_SIZE = 200
BLOCK_SIZE = 128
BATCH_SIZE = 32
LR = 5e-4
WEIGHT_DECAY = 0.01
MAX_TRAIN_SECONDS = 180
SEED = 0
VOCAB_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)


def encode_text(text: str) -> list[int]:
    return list(text.encode("utf-8", errors="ignore"))


def decode_tokens(tokens: list[int]) -> str:
    return bytes(tokens).decode("utf-8", errors="ignore")


def load_text_splits() -> tuple[list[str], list[str]]:
    dataset = load_dataset(DATASET_NAME, split="train")
    dataset = dataset.shuffle(seed=SEED)
    if len(dataset) < TRAIN_SIZE + VAL_SIZE:
        raise ValueError(f"Dataset has only {len(dataset)} examples, expected at least {TRAIN_SIZE + VAL_SIZE}.")
    val_texts = dataset.select(range(0, VAL_SIZE))["text"]
    train_texts = dataset.select(range(VAL_SIZE, VAL_SIZE + TRAIN_SIZE))["text"]
    return train_texts, val_texts


def build_token_stream(texts: list[str]) -> torch.Tensor:
    separator = [10]
    pieces: list[int] = []
    for text in texts:
        pieces.extend(encode_text(text))
        pieces.extend(separator)
    return torch.tensor(pieces, dtype=torch.long)


train_ids: torch.Tensor
val_ids: torch.Tensor


def initialize_data() -> None:
    global train_ids, val_ids
    train_texts, val_texts = load_text_splits()
    train_ids = build_token_stream(train_texts)
    val_ids = build_token_stream(val_texts)


initialize_data()


def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = train_ids if split == "train" else val_ids
    max_start = len(data) - BLOCK_SIZE - 1
    if max_start < 0:
        raise ValueError("Not enough tokens to build a batch. Reduce BLOCK_SIZE or increase data size.")
    starts = torch.randint(0, max_start + 1, (BATCH_SIZE,), device=DEVICE)
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in starts])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in starts])
    return x.to(DEVICE), y.to(DEVICE)


def evaluate_bpb(model: torch.nn.Module) -> float:
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(0, len(val_ids) - BLOCK_SIZE, BLOCK_SIZE):
            x = val_ids[i : i + BLOCK_SIZE].unsqueeze(0).to(DEVICE)
            y = val_ids[i + 1 : i + BLOCK_SIZE + 1].unsqueeze(0).to(DEVICE)
            logits = model(x)
            loss = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))
            total_nll += loss.item()
            total_tokens += BLOCK_SIZE
    if total_tokens == 0:
        raise ValueError("Validation data is too small for evaluation.")
    return total_nll / total_tokens / np.log(2)
