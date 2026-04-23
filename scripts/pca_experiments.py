import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from src.feature_loader import load_all_features
from src.models import run_all_models
from src.dim_reduction import reduce_tfidf, reduce_embeddings
from tqdm import tqdm


def make_json_safe(results):
    cleaned = []
    for r in results:
        r_copy = r.copy()

        # remove non-serializable
        r_copy.pop("fitted_model", None)

        # convert numpy → list
        if "confusion_matrix" in r_copy:
            r_copy["confusion_matrix"] = r_copy["confusion_matrix"].tolist()

        cleaned.append(r_copy)

    return cleaned


experiments, y_train, y_test = load_all_features("./features")

X_tfidf_train, X_tfidf_test = experiments["TF-IDF Bigram"]
X_emb_train, X_emb_test     = experiments["BERT Embedding"]


dims = [50, 100, 200]

pca_results = []

for d in tqdm(dims, desc="PCA Dimensions"):
    print(f"\nRunning experiments for dim = {d}")

    # ───── TF-IDF + SVD ─────
    X_tr, X_te, var = reduce_tfidf(X_tfidf_train, X_tfidf_test, d)

    res = run_all_models(
        X_tr, X_te, y_train, y_test,
        feature_name=f"TF-IDF SVD ({d})"
    )

    # attach explained variance
    for r in res:
        r["dim"] = d
        r["method"] = "TF-IDF SVD"
        r["explained_variance"] = float(var)

    pca_results.extend(res)

    print(f"TF-IDF SVD ({d}) → Explained Variance: {var:.4f}")


    # ───── BERT + PCA ─────
    X_tr, X_te, var = reduce_embeddings(X_emb_train, X_emb_test, d)

    res = run_all_models(
        X_tr, X_te, y_train, y_test,
        feature_name=f"BERT PCA ({d})"
    )

    # attach explained variance
    for r in res:
        r["dim"] = d
        r["method"] = "BERT PCA"
        r["explained_variance"] = float(var)

    pca_results.extend(res)

    print(f"BERT PCA ({d}) → Explained Variance: {var:.4f}")


safe_results = make_json_safe(pca_results)

with open("./pca_results.json", "w") as f:
    json.dump(safe_results, f, indent=2)

print("\nSaved pca_results.json")