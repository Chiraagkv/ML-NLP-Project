from sklearn.decomposition import TruncatedSVD, PCA

def reduce_tfidf(X_train, X_test, n_components=100):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_train_red = svd.fit_transform(X_train)
    X_test_red  = svd.transform(X_test)
    explained_var = svd.explained_variance_ratio_.sum()
    return X_train_red, X_test_red, explained_var


def reduce_embeddings(X_train, X_test, n_components=100):
    pca = PCA(n_components=n_components, random_state=42)
    X_train_red = pca.fit_transform(X_train)
    X_test_red  = pca.transform(X_test)
    explained_var = pca.explained_variance_ratio_.sum()
    return X_train_red, X_test_red, explained_var