from time import time
from sklearn.decomposition import PCA


def pca(X_train, X_test):
    n_components = 150
    pca_dr = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
    X_train_pca = pca_dr.transform(X_train)
    X_test_pca = pca_dr.transform(X_test)

    return X_train_pca, X_test_pca
