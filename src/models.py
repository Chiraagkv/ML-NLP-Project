"""
Training + evaluation 
Supports: Logistic Regression, Linear SVM,KNN, KMeans
Works with BoW, TF-IDF, Embeddings, PCA-reduced features.

Usage:
    from models.trainer import run_all_models
    results = run_all_models(X_train, X_test, y_train, y_test, feature_name="TF-IDF")
"""

import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, f1_score, classification_report, confusion_matrix,)
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def get_models():
    return {
        "Logistic Regression": LogisticRegression(
    max_iter=1000,
    solver="lbfgs",
    random_state=42,
),
        "Linear SVM": LinearSVC(C=1.0,max_iter=2000,random_state=42,),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5,metric="cosine",n_jobs=-1,),
        "KNN (k=11)": KNeighborsClassifier(n_neighbors=11,metric="cosine",n_jobs=-1,),
    }


def evaluate_kmeans(X_train, X_test, y_test, n_classes):
    # kmeans has no idea about labels so we fit it and then figure out
    # which cluster corresponds to which class after the fact
    km = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
    km.fit(X_train)
    clusters = km.predict(X_test)
 
    y = LabelEncoder().fit_transform(y_test)
    mapped = np.zeros_like(clusters)
    for cid in range(n_classes):
        idx = clusters == cid
        if idx.sum() == 0:
            continue
        # assign this cluster whatever label shows up most inside it
        mapped[idx] = np.bincount(y[idx]).argmax()
 
    return {
        "model":            "KMeans",
        "accuracy":         round(accuracy_score(y, mapped), 4),
        "macro_f1":         round(f1_score(y, mapped, average="macro", zero_division=0), 4),
        "report":           classification_report(y, mapped, zero_division=0),
        "confusion_matrix": confusion_matrix(y, mapped),
        "train_time_s":     None,
    }


def train_and_evaluate(name, model, X_train, X_test, y_train, y_test):
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - t0, 3)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    return {
        "model":            name,
        "accuracy":         round(acc, 4),
        "macro_f1":         round(f1, 4),
        "report":           report,
        "confusion_matrix": cm,
        "train_time_s":     train_time,
        "fitted_model":     model,   # keep for later inspection
    }

def run_all_models(X_train, X_test, y_train, y_test, feature_name="Features"):

    n_classes = len(np.unique(y_test))
    results   = []
    print(f"\n{'='*60}")
    print(f"  Feature Space : {feature_name}")
    print(f"  Train samples : {X_train.shape[0]}  |  Test samples : {X_test.shape[0]}")
    print(f"  Feature dim   : {X_train.shape[1]}  |  Classes      : {n_classes}")
    print(f"{'='*60}")
    print(f"  {'Model':<25} {'Accuracy':>10} {'Macro F1':>10} {'Time (s)':>10}")
    print(f"  {'-'*55}")

    # Supervised models
    for name, model in tqdm(get_models().items(), desc=f"{feature_name} Models" ):
        res = train_and_evaluate(name, model, X_train, X_test, y_train, y_test)
        results.append({**res, "feature_space": feature_name})
        print(f"  {res['model']:<25} {res['accuracy']:>10.4f} {res['macro_f1']:>10.4f} {res['train_time_s']:>10.3f}")

    # KMeans (unsupervised)
    km_res = evaluate_kmeans(X_train, X_test, y_test, n_classes)
    results.append({**km_res, "feature_space": feature_name})
    print(f"  {'KMeans':<25} {km_res['accuracy']:>10.4f} {km_res['macro_f1']:>10.4f} {'  N/A':>10}")

    print(f"{'='*60}\n")
    return results


def compare_feature_spaces(experiments: dict, y_train, y_test):

    all_results = []
    for feature_name, (X_train, X_test) in experiments.items():
        res = run_all_models(X_train, X_test, y_train, y_test, feature_name)
        all_results.extend(res)
    return all_results


# ─────────────────────────────────────────────
# Quick summary table (no matplotlib needed)
# ─────────────────────────────────────────────

def summary_table(all_results):
    print(f"\n{'='*75}")
    print(f"  {'Feature Space':<22} {'Model':<25} {'Accuracy':>10} {'Macro F1':>10}")
    print(f"  {'-'*70}")
    for r in sorted(all_results, key=lambda x: -x["macro_f1"]):
        print(
            f"  {r['feature_space']:<22} {r['model']:<25} "
            f"{r['accuracy']:>10.4f} {r['macro_f1']:>10.4f}"
        )
    print(f"{'='*75}\n")
