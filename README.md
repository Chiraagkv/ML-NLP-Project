# Sentiment Analysis with Feature Engineering and Embeddings

## Overview

This project explores multiple feature representations and machine learning models for sentiment classification on the IMDB dataset. The goal is to compare traditional text representations (BoW, TF-IDF) with pretrained embedding-based approaches and analyze their effectiveness.

---

## Project Structure

```
project/
│
├── features/              # Saved feature matrices (.npy/.npz/.pkl)
├── plots/                 # Generated plots
│
├── src/
│   ├── models.py
│   ├── feature_loader.py
│   ├── dim_reduction.py
│
├── scripts/
│   ├── train.py
│   ├── pca_experiments.py
│   ├── plot_results.py
│
├── notebooks/
│   ├── experiments.ipynb
│
├── all_results.json
├── pca_results.json
├── README.md
```

---

## Features Used

### 1. Bag of Words (BoW)

* Binary unigram representation
* Vocabulary pruning applied

### 2. TF-IDF

* Unigrams + bigrams
* Sublinear term frequency scaling
* Top features selected

### 3. Pretrained Embeddings

* BERT (DistilBERT)
* Mean pooling
* 768-dimensional dense vectors

---

## Models Evaluated

* Logistic Regression
* Linear SVM
* RBF SVM
* K-Nearest Neighbors
* K-Means (unsupervised baseline)

---

## Dimensionality Reduction

* TF-IDF → Truncated SVD
* Embeddings → PCA
* Tested dimensions: 50, 100, 200

---

## How to Run

From project root:

```bash
python scripts/train.py
python scripts/pca_experiments.py
python scripts/plot_results.py
```

---

## Key Results

* Best performance: **TF-IDF + SVM (~88% accuracy)**
* Embeddings perform competitively but slightly worse without fine-tuning
* Dimensionality reduction retains most performance with fewer features

---

## Key Insights

* **TF-IDF outperforms embeddings** for sentiment classification due to task-specific lexical signals
* **Linear models perform best**, indicating approximate linear separability
* **Embeddings are dense and highly similar**, reducing contrast between samples
* **PCA/SVD retains performance**, showing redundancy in high-dimensional features
* **t-SNE visualization shows class overlap**, confirming limited separability in embedding space

---

## Visualization Highlights

* Model performance comparison
* Accuracy vs dimensionality
* t-SNE projection of embeddings
* Discriminative TF-IDF features

---

## Engineering Practices

* Modular design (separate feature loading, models, experiments)
* Precomputed features (no redundant computation)
* Command-line interface using argparse
* Progress tracking with tqdm
* Clean separation of training, experimentation, and plotting
#### Features

Precomputed features are not included due to size.

To regenerate:
- Run `notebooks/experiments.ipynb`

---

## Conclusion

Traditional sparse representations (TF-IDF) remain highly effective for sentiment analysis, outperforming pretrained embeddings when used without fine-tuning. Dimensionality reduction significantly compresses feature space with minimal performance loss, highlighting redundancy in both sparse and dense representations.

---

## Future Work

* Fine-tuning transformer models
* Exploring ensemble methods
* Using more advanced sentence embeddings
* Hyperparameter optimization

---
