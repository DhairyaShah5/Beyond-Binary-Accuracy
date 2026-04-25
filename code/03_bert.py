"""
BERT Fine-tuning for IMDb Sentiment Classification
=====================================================
Fine-tunes bert-base-uncased on the IMDb training set and evaluates
on the full test set + challenging slices.

Architecture:
  - bert-base-uncased (110M parameters)
  - [CLS] token representation → Dropout → Linear → Sigmoid
  - Max sequence length: 512 tokens (BERT's limit)

Training:
  - AdamW optimizer with weight decay 0.01
  - Linear warmup schedule (10% of steps) + linear decay
  - 3 epochs, batch size 16 (with gradient accumulation if needed)
  - Learning rate: 2e-5

Note:
  Reviews longer than 512 tokens are truncated. This is a known
  limitation documented in the project report — BERT cannot see
  the full text of long reviews.

Usage:
  python 03_bert.py                    # train + evaluate
  python 03_bert.py --epochs 4         # override epochs
  python 03_bert.py --batch-size 8 --grad-accum 4   # for limited GPU memory
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Add project root so we can import shared utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_data import (
    load_imdb, clean, build_slices,
    evaluate_on_slices, print_results,
)

# ------------------------------------------------------------------ #
# Config & CLI
# ------------------------------------------------------------------ #
parser = argparse.ArgumentParser(description="BERT fine-tuning for IMDb sentiment")
parser.add_argument("--model-name", type=str, default="bert-base-uncased")
parser.add_argument("--max-len", type=int, default=512,
                    help="Max token length (BERT limit is 512)")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--grad-accum", type=int, default=1,
                    help="Gradient accumulation steps (increase if OOM)")
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--warmup-ratio", type=float, default=0.1)
parser.add_argument("--weight-decay", type=float, default=0.01)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)
np.random.seed(args.seed)
print(f"Device: {DEVICE}")

OUT_CSV = os.path.join(os.path.dirname(__file__), "03_bert_results.csv")

# ------------------------------------------------------------------ #
# 1. Load data
# ------------------------------------------------------------------ #
print("Loading IMDb data ...")
t0 = time.time()
X_train_text, y_train, X_test_text, y_test = load_imdb()
print(f"  train: {len(X_train_text)}  test: {len(X_test_text)}  ({time.time()-t0:.1f}s)")

# Clean text for slice construction (slices use cleaned text)
print("Cleaning text for slice construction ...")
X_test_clean = [clean(t) for t in X_test_text]

# ------------------------------------------------------------------ #
# 2. Tokenizer
# ------------------------------------------------------------------ #
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup

print(f"Loading tokenizer: {args.model_name} ...")
tokenizer = BertTokenizer.from_pretrained(args.model_name)

# ------------------------------------------------------------------ #
# 3. Dataset
# ------------------------------------------------------------------ #
class IMDbBertDataset(Dataset):
    """Tokenizes raw review text on-the-fly for BERT."""

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# Note: we pass RAW text to BERT (not our cleaned text), because
# BERT's own tokenizer handles casing, subword splitting, etc.
train_ds = IMDbBertDataset(X_train_text, y_train, tokenizer, args.max_len)
test_ds = IMDbBertDataset(X_test_text, y_test, tokenizer, args.max_len)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

# ------------------------------------------------------------------ #
# 4. Model
# ------------------------------------------------------------------ #
print(f"Loading model: {args.model_name} ...")
model = BertForSequenceClassification.from_pretrained(
    args.model_name, num_labels=2,
)
model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters:     {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

# ------------------------------------------------------------------ #
# 5. Optimizer & scheduler
# ------------------------------------------------------------------ #
optimizer = torch.optim.AdamW(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
)

total_steps = (len(train_loader) // args.grad_accum) * args.epochs
warmup_steps = int(total_steps * args.warmup_ratio)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
)

print(f"\nTotal training steps: {total_steps}  (warmup: {warmup_steps})")

# ------------------------------------------------------------------ #
# 6. Training loop
# ------------------------------------------------------------------ #
print(f"\nTraining for {args.epochs} epochs ...")
for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    t0 = time.time()
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader, 1):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss / args.grad_accum
        loss.backward()

        total_loss += outputs.loss.item() * input_ids.size(0)
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += input_ids.size(0)

        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Progress logging
        if step % 100 == 0:
            print(f"    step {step}/{len(train_loader)}  "
                  f"loss={total_loss/total:.4f}  acc={correct/total:.4f}")

    train_acc = correct / total
    avg_loss = total_loss / total

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += input_ids.size(0)

    val_acc = val_correct / val_total

    print(f"  Epoch {epoch}/{args.epochs}  "
          f"loss={avg_loss:.4f}  train_acc={train_acc:.4f}  "
          f"val_acc={val_acc:.4f}  ({time.time()-t0:.1f}s)")

# ------------------------------------------------------------------ #
# 7. Final evaluation on all slices
# ------------------------------------------------------------------ #
print("\nFinal evaluation on test set + challenging slices ...")

model.eval()
all_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())

y_pred = np.array(all_preds)
y_true = np.array(y_test)

# Build slices from cleaned test text
slices = build_slices(X_test_clean)

# Evaluate
results = evaluate_on_slices(y_true, y_pred, slices)

print("\n=== BERT RESULTS ===")
df = print_results(results, model_name=f"BERT ({args.model_name})")

# Save CSV
df.to_csv(OUT_CSV, index=False)
print(f"\nResults saved to: {OUT_CSV}")
