"""
CNN Sentiment Classifier — Kim (2014) Architecture
====================================================
Multi-filter CNN over pre-trained GloVe embeddings for IMDb sentiment
classification, evaluated on the full test set and challenging slices.

Architecture:
  - Embedding layer initialized with GloVe 6B 100d vectors
  - Parallel 1D convolutions with filter widths 3, 4, 5 (128 filters each)
  - Max-over-time pooling per filter bank
  - Concatenation → Dropout (0.5) → Fully-connected → Sigmoid

Training:
  - Adam optimizer, lr=1e-3
  - Binary cross-entropy loss
  - 5 epochs, batch size 64
  - Embedding layer is fine-tuned during training

Usage:
  python 02_cnn.py                   # train + evaluate
  python 02_cnn.py --epochs 10       # override epochs
  python 02_cnn.py --glove-path /path/to/glove.6B.100d.txt
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# Add project root so we can import shared utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_data import (
    load_imdb, clean, build_slices,
    evaluate_on_slices, print_results, PROJECT_ROOT,
)

# ------------------------------------------------------------------ #
# Config & CLI
# ------------------------------------------------------------------ #
parser = argparse.ArgumentParser(description="CNN sentiment classifier (Kim 2014)")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--max-len", type=int, default=500,
                    help="Max sequence length in tokens (pad/truncate)")
parser.add_argument("--max-vocab", type=int, default=50000)
parser.add_argument("--embed-dim", type=int, default=100)
parser.add_argument("--num-filters", type=int, default=128)
parser.add_argument("--filter-sizes", type=int, nargs="+", default=[3, 4, 5])
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--glove-path", type=str, default=None,
                    help="Path to glove.6B.100d.txt (auto-downloads if missing)")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)
np.random.seed(args.seed)
print(f"Device: {DEVICE}")

OUT_CSV = os.path.join(os.path.dirname(__file__), "02_cnn_results.csv")

# ------------------------------------------------------------------ #
# 1. Load & preprocess data
# ------------------------------------------------------------------ #
print("Loading IMDb data ...")
t0 = time.time()
X_train_text, y_train, X_test_text, y_test = load_imdb()
print(f"  train: {len(X_train_text)}  test: {len(X_test_text)}  ({time.time()-t0:.1f}s)")

print("Cleaning text ...")
X_train_clean = [clean(t) for t in X_train_text]
X_test_clean = [clean(t) for t in X_test_text]

# ------------------------------------------------------------------ #
# 2. Build vocabulary
# ------------------------------------------------------------------ #
print("Building vocabulary ...")
counter = Counter()
for text in X_train_clean:
    counter.update(text.split())

# Reserve 0=pad, 1=unk
vocab = {word: idx + 2 for idx, (word, _) in
         enumerate(counter.most_common(args.max_vocab))}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1
VOCAB_SIZE = len(vocab)
print(f"  Vocabulary size: {VOCAB_SIZE}")


def text_to_indices(text, max_len):
    """Convert cleaned text to padded/truncated index sequence."""
    words = text.split()[:max_len]
    indices = [vocab.get(w, 1) for w in words]
    # Pad
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    return indices

# ------------------------------------------------------------------ #
# 3. Load GloVe embeddings
# ------------------------------------------------------------------ #
def download_glove(dest_dir):
    """Download and extract GloVe 6B if not present."""
    import zipfile
    import urllib.request

    zip_path = os.path.join(dest_dir, "glove.6B.zip")
    txt_path = os.path.join(dest_dir, "glove.6B.100d.txt")

    if os.path.exists(txt_path):
        return txt_path

    os.makedirs(dest_dir, exist_ok=True)
    url = "https://nlp.stanford.edu/data/glove.6B.zip"
    print(f"  Downloading GloVe from {url} ...")
    print("  (This is ~862 MB and may take a few minutes)")
    urllib.request.urlretrieve(url, zip_path)

    print("  Extracting glove.6B.100d.txt ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract("glove.6B.100d.txt", dest_dir)

    os.remove(zip_path)
    return txt_path


def load_glove(glove_path, embed_dim):
    """Load GloVe vectors and build an embedding matrix aligned to our vocab."""
    print(f"Loading GloVe from {glove_path} ...")
    glove = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in vocab:
                glove[word] = np.array(parts[1:], dtype=np.float32)

    # Initialize embedding matrix
    embedding_matrix = np.random.uniform(-0.25, 0.25, (VOCAB_SIZE, embed_dim)).astype(np.float32)
    embedding_matrix[0] = 0  # PAD
    hits = 0
    for word, idx in vocab.items():
        if word in glove:
            embedding_matrix[idx] = glove[word]
            hits += 1

    print(f"  GloVe coverage: {hits}/{VOCAB_SIZE} vocab words ({hits/VOCAB_SIZE*100:.1f}%)")
    return embedding_matrix


# Resolve GloVe path
if args.glove_path and os.path.exists(args.glove_path):
    glove_path = args.glove_path
else:
    glove_dir = os.path.join(PROJECT_ROOT, "data", "glove")
    glove_path = os.path.join(glove_dir, "glove.6B.100d.txt")
    if not os.path.exists(glove_path):
        glove_path = download_glove(glove_dir)

embedding_matrix = load_glove(glove_path, args.embed_dim)

# ------------------------------------------------------------------ #
# 4. Dataset & DataLoader
# ------------------------------------------------------------------ #
class IMDbDataset(Dataset):
    def __init__(self, texts, labels, max_len):
        self.data = [text_to_indices(t, max_len) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


train_ds = IMDbDataset(X_train_clean, y_train, args.max_len)
test_ds = IMDbDataset(X_test_clean, y_test, args.max_len)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

# ------------------------------------------------------------------ #
# 5. CNN Model
# ------------------------------------------------------------------ #
class TextCNN(nn.Module):
    """Kim (2014) multi-filter CNN for text classification."""

    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes,
                 dropout, pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(
                torch.from_numpy(pretrained_embeddings)
            )

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), 1)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)          # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)         # (batch, embed_dim, seq_len)

        conv_outs = []
        for conv in self.convs:
            c = torch.relu(conv(x))    # (batch, num_filters, seq_len - fs + 1)
            c = c.max(dim=2).values    # max-over-time pooling → (batch, num_filters)
            conv_outs.append(c)

        x = torch.cat(conv_outs, dim=1)  # (batch, num_filters * len(filter_sizes))
        x = self.dropout(x)
        x = self.fc(x).squeeze(1)        # (batch,)
        return x


model = TextCNN(
    vocab_size=VOCAB_SIZE,
    embed_dim=args.embed_dim,
    num_filters=args.num_filters,
    filter_sizes=args.filter_sizes,
    dropout=args.dropout,
    pretrained_embeddings=embedding_matrix,
).to(DEVICE)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# ------------------------------------------------------------------ #
# 6. Training
# ------------------------------------------------------------------ #
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.BCEWithLogitsLoss()

print(f"\nTraining for {args.epochs} epochs ...")
for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    t0 = time.time()

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        preds = (logits > 0).long()
        correct += (preds == batch_y.long()).sum().item()
        total += batch_x.size(0)

    train_acc = correct / total
    avg_loss = total_loss / total

    # Quick validation accuracy
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            logits = model(batch_x)
            preds = (logits > 0).long()
            val_correct += (preds == batch_y.long()).sum().item()
            val_total += batch_x.size(0)
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
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(DEVICE)
        logits = model(batch_x)
        preds = (logits > 0).long()
        all_preds.extend(preds.cpu().numpy())

y_pred = np.array(all_preds)
y_true = np.array(y_test)

# Build slices
slices = build_slices(X_test_clean)

# Evaluate
results = evaluate_on_slices(y_true, y_pred, slices)

print("\n=== CNN RESULTS ===")
df = print_results(results, model_name="CNN (GloVe + Kim2014)")

# Save CSV
df.to_csv(OUT_CSV, index=False)
print(f"\nResults saved to: {OUT_CSV}")

# ------------------------------------------------------------------ #
# 8. SST-2 SECONDARY BENCHMARK
# ------------------------------------------------------------------ #
print("\n" + "="*60)
print("SST-2 SECONDARY BENCHMARK")
print("="*60)

from utils_data import load_sst2

X_train_sst, y_train_sst, X_test_sst, y_test_sst = load_sst2()
X_train_sst_clean = [clean(t) for t in X_train_sst]
X_test_sst_clean  = [clean(t) for t in X_test_sst]

# Build vocab from SST-2 training data (reuse IMDb vocab for embeddings)
# SST-2 sentences are short, so we use a shorter max_len
SST_MAX_LEN = 64

sst_train_ds = IMDbDataset(X_train_sst_clean, y_train_sst, SST_MAX_LEN)
sst_test_ds  = IMDbDataset(X_test_sst_clean, y_test_sst, SST_MAX_LEN)
sst_train_loader = DataLoader(sst_train_ds, batch_size=args.batch_size, shuffle=True)
sst_test_loader  = DataLoader(sst_test_ds, batch_size=args.batch_size, shuffle=False)

# Train a fresh CNN on SST-2
print("Training CNN on SST-2 ...")
sst_model = TextCNN(
    vocab_size=VOCAB_SIZE,
    embed_dim=args.embed_dim,
    num_filters=args.num_filters,
    filter_sizes=args.filter_sizes,
    dropout=args.dropout,
    pretrained_embeddings=embedding_matrix,
).to(DEVICE)

sst_optimizer = torch.optim.Adam(sst_model.parameters(), lr=args.lr)

for epoch in range(1, args.epochs + 1):
    sst_model.train()
    total_loss = 0
    correct = 0
    total = 0
    t0 = time.time()

    for batch_x, batch_y in sst_train_loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        sst_optimizer.zero_grad()
        logits = sst_model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        sst_optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        preds = (logits > 0).long()
        correct += (preds == batch_y.long()).sum().item()
        total += batch_x.size(0)

    train_acc = correct / total

    sst_model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_x, batch_y in sst_test_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            logits = sst_model(batch_x)
            preds = (logits > 0).long()
            val_correct += (preds == batch_y.long()).sum().item()
            val_total += batch_x.size(0)
    val_acc = val_correct / val_total

    print(f"  Epoch {epoch}/{args.epochs}  "
          f"loss={total_loss/total:.4f}  train_acc={train_acc:.4f}  "
          f"val_acc={val_acc:.4f}  ({time.time()-t0:.1f}s)")

# Final SST-2 evaluation
sst_model.eval()
sst_preds = []
with torch.no_grad():
    for batch_x, batch_y in sst_test_loader:
        batch_x = batch_x.to(DEVICE)
        logits = sst_model(batch_x)
        preds = (logits > 0).long()
        sst_preds.extend(preds.cpu().numpy())

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_sst_pred = np.array(sst_preds)
y_sst_true = np.array(y_test_sst)

sst_results = [{
    "dataset": "SST-2",
    "model": "CNN (GloVe + Kim2014)",
    "n": len(y_sst_true),
    "acc": accuracy_score(y_sst_true, y_sst_pred),
    "prec": precision_score(y_sst_true, y_sst_pred, zero_division=0),
    "rec": recall_score(y_sst_true, y_sst_pred, zero_division=0),
    "f1": f1_score(y_sst_true, y_sst_pred, zero_division=0),
}]

import pandas as pd
sst_df = pd.DataFrame(sst_results)
print("\n=== CNN SST-2 RESULTS ===")
print(sst_df.to_string(index=False,
      formatters={"acc":"{:.4f}".format, "prec":"{:.4f}".format,
                  "rec":"{:.4f}".format, "f1":"{:.4f}".format}))

OUT_SST_CSV = os.path.join(os.path.dirname(__file__), "02_cnn_sst2_results.csv")
sst_df.to_csv(OUT_SST_CSV, index=False)
print(f"\nSST-2 results saved to: {OUT_SST_CSV}")
