# Beyond Binary Accuracy

**Evaluating Long-form Context and Challenging Cases in Movie Review Sentiment Analysis**

USC CSCI-544 Applied Natural Language Processing - Group 48

Dhairya Shah, Raghav Sarmukaddam, Jimmy Taravia, Yash Desai, Vivek Lakhani

---

## Overview

Standard sentiment analysis benchmarks report a single accuracy number, but that number is dominated by easy examples. This project investigates whether newer model generations (classical ML, CNN, BERT) genuinely improve on *difficult* reviews - long narratives, negation-heavy text, and mixed-sentiment opinions - or just get better at easy ones.

We train five models across three generations and evaluate each on the full IMDb test set **and** on three challenging subsets (slices) to expose where accuracy alone is misleading.

## Results Summary

### IMDb (Full Test Set)

| Model | Accuracy | F1 |
|---|---|---|
| Naive Bayes (TF-IDF) | 85.73% | 85.40% |
| Linear SVM (TF-IDF) | 88.10% | 88.01% |
| Hybrid SVM (TF-IDF + lexicon) | 88.29% | 88.19% |
| CNN (GloVe + Kim 2014) | 85.40% | 85.17% |
| BERT (bert-base-uncased) | 94.18% | 94.20% |

### F1 Drop on Mixed-Sentiment Slice

| Model | Full F1 | Mixed-Sentiment F1 | Drop |
|---|---|---|---|
| Hybrid SVM | 88.19% | 76.75% | -11.4 pp |
| CNN | 85.17% | 71.30% | -13.9 pp |
| BERT | 94.20% | 88.18% | -6.0 pp |

Even BERT, at 94% overall accuracy, drops 6 percentage points on mixed-sentiment reviews - confirming that accuracy alone does not capture model robustness.

## Repository Structure

```
.
├── code/
│   ├── utils_data.py                    # Shared data loading, preprocessing, slices, evaluation
│   ├── 01_baselines.py                  # Naive Bayes, SVM, Hybrid SVM (TF-IDF)
│   ├── 01_baselines.ipynb               # Notebook version of baselines
│   ├── 02_cnn.py                        # CNN (Kim 2014) standalone script
│   ├── 03_bert.py                       # BERT fine-tuning standalone script
│   ├── 04_colab_train_cnn_bert.ipynb    # Colab notebook for CNN + BERT (GPU required)
│   ├── 05_trained_notebook.ipynb        # Pre-run notebook with all outputs and results
│   ├── 01_baselines_results.csv         # Baseline results on IMDb
│   ├── 01_baselines_sst2_results.csv    # Baseline results on SST-2
│   ├── 02_cnn_results.csv               # CNN results on IMDb
│   ├── 02_cnn_sst2_results.csv          # CNN results on SST-2
│   ├── 03_bert_results.csv              # BERT results on IMDb
│   ├── 03_bert_sst2_results.csv         # BERT results on SST-2
│   └── combined_cnn_bert_results.csv    # CNN + BERT side-by-side
├── data/                                # Datasets (not tracked - see setup below)
├── requirements.txt
└── README.md
```

## Environment Setup

### Requirements

- Python 3.10+
- For baselines: CPU is sufficient
- For CNN and BERT: NVIDIA GPU recommended (we used Google Colab Pro with an A100 80GB)

### Installation

```bash
git clone https://github.com/DhairyaShah5/Beyond-Binary-Accuracy.git
cd Beyond-Binary-Accuracy
pip install -r requirements.txt
```

### Data Setup

**IMDb** - download and extract into `data/`:
```bash
cd data
wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz
rm aclImdb_v1.tar.gz
```

**SST-2** - downloaded automatically via HuggingFace `datasets` library when the scripts run.

**GloVe** (for CNN only) - downloaded automatically by `02_cnn.py` on first run, or manually:
```bash
mkdir -p data/glove && cd data/glove
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip glove.6B.100d.txt
rm glove.6B.zip
```

## Running the Code

### 1. Baselines (CPU, ~2 minutes)

```bash
python code/01_baselines.py
```

Trains Naive Bayes, Linear SVM, and Hybrid SVM on IMDb using TF-IDF features. Evaluates on the full test set, three challenging slices, and SST-2. Outputs results to `code/01_baselines_results.csv` and `code/01_baselines_sst2_results.csv`.

### 2. CNN + BERT (GPU required)

**Option A - Google Colab (recommended):**

Open `code/04_colab_train_cnn_bert.ipynb` in Google Colab, set runtime to GPU (T4 or A100), and run all cells. The notebook is self-contained - it downloads all data, trains both models, and lets you download the result CSVs at the end.

If you just want to see the outputs without re-running, `code/05_trained_notebook.ipynb` contains the pre-run notebook with all training logs, metrics, and results already populated.

**Option B - Local GPU:**

```bash
python code/02_cnn.py          # CNN (Kim 2014 + GloVe)
python code/03_bert.py         # BERT fine-tuning
```

Both scripts accept command-line arguments (run with `--help` to see options).

### Approximate Runtimes

| Model | CPU | T4 GPU | A100 GPU |
|---|---|---|---|
| Baselines | ~2 min | - | - |
| CNN (10 epochs) | ~45 min | ~8 min | ~3 min |
| BERT (4 epochs, IMDb) | impractical | ~120 min | ~45 min |
| BERT (4 epochs, SST-2) | impractical | ~80 min | ~26 min |

## How Results Are Generated

1. **Preprocessing**: All text goes through the same pipeline - strip HTML, lowercase, remove non-alphabetic characters, collapse whitespace. BERT uses its own WordPiece tokenizer on raw text instead.

2. **Features**: TF-IDF (unigram + bigram, 50k features, sublinear TF) for baselines. GloVe 100d embeddings for CNN. BERT uses its pre-trained tokenizer.

3. **Challenging Slices**: Three subsets of the IMDb test set, defined by heuristics applied to cleaned text:
   - **Long reviews**: >= 500 words
   - **Negation-heavy**: >= 2 negation cues (not, never, no, hardly, etc.)
   - **Mixed sentiment**: >= 2 positive AND >= 2 negative lexicon hits

4. **Metrics**: Accuracy, Precision, Recall, and F1 are computed for every model on every slice using scikit-learn.

5. **SST-2**: Each model is separately trained and evaluated on SST-2 (Stanford Sentiment Treebank) as a secondary benchmark for phrase-level vs. document-level comparison. The validation split is used as the test set since official test labels are hidden.

## Device Information

- **Baselines**: Trained locally on macOS (Apple Silicon, CPU only)
- **CNN and BERT**: Trained on Google Colab Pro - NVIDIA A100-SXM4-80GB, CUDA, PyTorch
