from sklearn.neural_network import MLPClassifier
from time import time


def mlp_train(X_train_pca, y_train):
    clf_mlp = MLPClassifier(activation='logistic', max_iter=500)
    t0 = time()
    clf_mlp.fit(X_train_pca, y_train)
    tn = time() - t0
    print("Training the MLP Classifier done in:", tn)
    return clf_mlp, tn


def mlp_predict(cld_mlp, X_test):
    t0 = time()
    y_pred = cld_mlp.predict(X_test)
    tn = time() - t0
    print("Predicting using MLP Classifier done in:", tn)
    return y_pred, tn
