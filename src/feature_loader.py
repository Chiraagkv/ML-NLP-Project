import numpy as np
import pickle
from scipy.sparse import load_npz
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────
# LABELS
# ─────────────────────────────────────────────────────────────

def load_labels(path="."):
    y_train = np.load(f"{path}/y_train.npy")
    y_test  = np.load(f"{path}/y_test.npy")
    return y_train, y_test


# ─────────────────────────────────────────────────────────────
# SPARSE FEATURES (BoW / TF-IDF)
# ─────────────────────────────────────────────────────────────

def load_bow(path="."):
    X_train = load_npz(f"{path}/X_bow_train.npz")
    X_test  = load_npz(f"{path}/X_bow_test.npz")
    return X_train, X_test


def load_tfidf(path="."):
    X_train = load_npz(f"{path}/X_tfidf_train.npz")
    X_test  = load_npz(f"{path}/X_tfidf_test.npz")
    return X_train, X_test


# ─────────────────────────────────────────────────────────────
# EMBEDDINGS (dense)
# ─────────────────────────────────────────────────────────────

def load_embeddings(path=".", scale=True):
    X_train = np.load(f"{path}/X_train_emb_mean.npy")
    X_test  = np.load(f"{path}/X_test_emb_mean.npy")

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

    return X_train, X_test


# ─────────────────────────────────────────────────────────────
# VECTORIZERS (optional)
# ─────────────────────────────────────────────────────────────

def load_vectorizers(path="."):
    with open(f"{path}/bow_vectorizer.pkl", "rb") as f:
        bow_vec = pickle.load(f)

    with open(f"{path}/tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vec = pickle.load(f)

    return bow_vec, tfidf_vec


# ─────────────────────────────────────────────────────────────
# FEATURE NAMES (optional)
# ─────────────────────────────────────────────────────────────

def load_tfidf_feature_names(path="."):
    return np.load(f"{path}/tfidf_feature_names.npy", allow_pickle=True)


# ─────────────────────────────────────────────────────────────
# ALL FEATURES (MAIN ENTRY POINT)
# ─────────────────────────────────────────────────────────────

def load_all_features(path=".", scale_embeddings=True):
    """
    Returns:
        experiments dict ready for trainer.py
        y_train, y_test
    """

    y_train, y_test = load_labels(path)

    X_bow_train, X_bow_test       = load_bow(path)
    X_tfidf_train, X_tfidf_test   = load_tfidf(path)
    X_emb_train, X_emb_test       = load_embeddings(path, scale=scale_embeddings)

    experiments = {
        "BoW Unigram":   (X_bow_train,   X_bow_test),
        "TF-IDF Bigram": (X_tfidf_train, X_tfidf_test),
        "BERT Embedding": (X_emb_train,  X_emb_test),
    }

    return experiments, y_train, y_test