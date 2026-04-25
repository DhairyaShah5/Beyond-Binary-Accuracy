"""
Shared data utilities for the Beyond Binary Accuracy project.
==============================================================
Provides consistent data loading, preprocessing, lexicon definitions,
and challenging-slice construction used by all model scripts
(01_baselines, 02_cnn, 03_bert).
"""

import os
import re
import glob
import numpy as np

# ------------------------------------------------------------------ #
# Paths
# ------------------------------------------------------------------ #
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "aclImdb")

# ------------------------------------------------------------------ #
# Sentiment lexicons
# ------------------------------------------------------------------ #
POS_WORDS = set("""
good great excellent amazing wonderful fantastic brilliant awesome loved
love enjoyable enjoy best perfect beautiful fun funny hilarious superb
outstanding masterpiece memorable touching heartwarming powerful charming
delightful impressive engaging gripping thrilling stunning magnificent
""".split())

NEG_WORDS = set("""
bad terrible awful horrible worst boring dull stupid poor waste wasted
disappointing disappointed hate hated hates annoying pathetic lame ridiculous
worse lousy mediocre forgettable painful laughable weak predictable cliched
unwatchable tedious bland flat sloppy uninspired cringe cringy nonsense
""".split())

NEGATIONS = set("""
not no never none nothing neither nor hardly scarcely barely without cannot
cant couldnt wasnt werent isnt arent dont didnt doesnt wont wouldnt shouldnt
""".split())

# ------------------------------------------------------------------ #
# Data loading
# ------------------------------------------------------------------ #
def load_split(split):
    """Load one split ('train' or 'test') from the IMDb directory.

    Returns:
        texts: list[str]  — raw review strings
        labels: list[int] — 1 = positive, 0 = negative
    """
    texts, labels = [], []
    for label_name, label_id in [("pos", 1), ("neg", 0)]:
        pattern = os.path.join(DATA_DIR, split, label_name, "*.txt")
        for path in sorted(glob.glob(pattern)):
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())
            labels.append(label_id)
    return texts, labels


def load_imdb():
    """Load full IMDb dataset.

    Returns:
        (X_train_text, y_train, X_test_text, y_test)
    """
    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")
    return X_train, y_train, X_test, y_test

# ------------------------------------------------------------------ #
# Preprocessing
# ------------------------------------------------------------------ #
_HTML_TAG = re.compile(r"<[^>]+>")
_NON_ALPHA = re.compile(r"[^a-z\s]")


def clean(text):
    """Strip HTML, lowercase, remove non-alpha, collapse whitespace."""
    text = _HTML_TAG.sub(" ", text)
    text = text.lower()
    text = _NON_ALPHA.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ------------------------------------------------------------------ #
# Challenging slices
# ------------------------------------------------------------------ #
def build_slices(cleaned_texts):
    """Build challenging-slice index lists from cleaned test texts.

    Thresholds (matching project reports):
        - Long reviews:    >= 500 words
        - Negation-heavy:  >= 2 negation cues
        - Mixed sentiment: >= 2 positive AND >= 2 negative lexicon hits

    Returns:
        dict mapping slice_name -> list[int] of indices
    """
    long_idx, neg_idx, mixed_idx = [], [], []

    for i, t in enumerate(cleaned_texts):
        words = t.split()
        wc = len(words)

        # Long reviews
        if wc >= 500:
            long_idx.append(i)

        # Negation-heavy
        neg_cues = sum(w in NEGATIONS for w in words)
        if neg_cues >= 2:
            neg_idx.append(i)

        # Mixed sentiment
        pos_hits = sum(w in POS_WORDS for w in words)
        neg_hits = sum(w in NEG_WORDS for w in words)
        if pos_hits >= 2 and neg_hits >= 2:
            mixed_idx.append(i)

    slices = {
        "Full test set": list(range(len(cleaned_texts))),
        f"Long reviews (>=500 words) [n={len(long_idx)}]": long_idx,
        f"Negation-heavy (>=2 cues) [n={len(neg_idx)}]": neg_idx,
        f"Mixed sentiment (pos&neg>=2) [n={len(mixed_idx)}]": mixed_idx,
    }
    return slices

# ------------------------------------------------------------------ #
# Evaluation helpers
# ------------------------------------------------------------------ #
def evaluate_on_slices(y_true, y_pred, slices):
    """Compute Accuracy, Precision, Recall, F1 per slice.

    Args:
        y_true:  np.ndarray of ground-truth labels
        y_pred:  np.ndarray of predicted labels
        slices:  dict from build_slices()

    Returns:
        list[dict] with keys: slice, n, acc, prec, rec, f1
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
    )

    results = []
    for slice_name, idx in slices.items():
        idx = np.array(idx, dtype=int)
        yt = y_true[idx]
        yp = y_pred[idx]
        results.append({
            "slice": slice_name,
            "n": len(idx),
            "acc": accuracy_score(yt, yp),
            "prec": precision_score(yt, yp, zero_division=0),
            "rec": recall_score(yt, yp, zero_division=0),
            "f1": f1_score(yt, yp, zero_division=0),
        })
    return results


def print_results(results, model_name=""):
    """Pretty-print evaluation results."""
    import pandas as pd

    df = pd.DataFrame(results)
    if model_name:
        df.insert(0, "model", model_name)
    print(df.to_string(
        index=False,
        formatters={
            "acc": "{:.4f}".format,
            "prec": "{:.4f}".format,
            "rec": "{:.4f}".format,
            "f1": "{:.4f}".format,
        },
    ))
    return df
