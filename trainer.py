# ─────────────────────────────────────────────────────────────
# CELL 1 — imports
# ─────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from Models import compare_feature_spaces, summary_table
from datasets import load_dataset
dataset = load_dataset("imdb")


# ─────────────────────────────────────────────────────────────
# CELL 2 — sample 8k and split
# ─────────────────────────────────────────────────────────────

# imdb comes with a fixed train/test split, we merge and resample
# so we control the exact 8k and get a clean 80/20 split
all_texts  = dataset["train"]["text"] + dataset["test"]["text"]
all_labels = dataset["train"]["label"] + dataset["test"]["label"]

# sample exactly 8000
np.random.seed(42)
idx = np.random.choice(len(all_texts), size=8000, replace=False)
texts  = [all_texts[i]  for i in idx]
labels = [all_labels[i] for i in idx]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Train: {len(X_train_raw)} | Test: {len(X_test_raw)}")
print(f"Classes: {set(labels)}")   # 0 = neg, 1 = pos


# ─────────────────────────────────────────────────────────────
# CELL 3 — build features
# ─────────────────────────────────────────────────────────────

# BoW — unigram, binary counts, prune vocab to top 20k
bow_vec = CountVectorizer(
    binary=True,
    max_features=20000,
    stop_words="english",
    ngram_range=(1, 1),
)
X_bow_train = bow_vec.fit_transform(X_train_raw)
X_bow_test  = bow_vec.transform(X_test_raw)
print(f"BoW shape: {X_bow_train.shape}")

# TF-IDF — bigram, prune to top 30k
tfidf_vec = TfidfVectorizer(
    max_features=30000,
    stop_words="english",
    ngram_range=(1, 2),   # unigrams + bigrams
    sublinear_tf=True,    # apply log normalization on tf
)
X_tfidf_train = tfidf_vec.fit_transform(X_train_raw)
X_tfidf_test  = tfidf_vec.transform(X_test_raw)
print(f"TF-IDF shape: {X_tfidf_train.shape}")


# ─────────────────────────────────────────────────────────────
# CELL 4 — train everything and print results
# ─────────────────────────────────────────────────────────────

experiments = {
    "BoW Unigram":   (X_bow_train,   X_bow_test),
    "TF-IDF Bigram": (X_tfidf_train, X_tfidf_test),
}

all_results = compare_feature_spaces(experiments, y_train, y_test)

# sorted leaderboard
summary_table(all_results)