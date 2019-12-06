from sklearn.svm import SVC
from time import time


def svm_train(X_train_pca, y_train):
    clf_svm = SVC(C=1000, gamma=0.005)
    t0 = time()
    clf_svm.fit(X_train_pca, y_train)
    tn = time() - t0
    print("\nTraining the SVM Classifier done in:", tn)
    return clf_svm, tn


def svm_predict(clf_svm, X_test_pca):
    t0 = time()
    y_pred = clf_svm.predict(X_test_pca)
    tn = time() - t0
    print("\nPredicting using SVM Classifier done in:", tn)
    return y_pred, tn
