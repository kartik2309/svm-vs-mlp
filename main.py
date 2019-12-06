from get_lfw_faces import get_lfw_faces
from mlp import mlp_predict, mlp_train
from svm import svm_predict, svm_train
from pca import pca

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt


def comparison(mnfpp):
    X, y = get_lfw_faces(mnfpp)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                        random_state=42)

    # Dimensionality reduction using PCA
    X_train_pca, X_test_pca = pca(X_train, X_test)
    trainset_size = X_test_pca.shape[0]
    testset_size = X_test_pca.shape[0]

    # Training SVM Classifier
    clf_svm, tn_svm_train = svm_train(X_train_pca, y_train)

    # Training MLP Classifier
    clf_mlp, tn_mlp_train = mlp_train(X_train_pca, y_train)

    # Predicting using SVM Classifier
    y_pred_svm, tn_svm_predict = svm_predict(clf_svm, X_test_pca)

    # Predicting using MLP Classifier
    y_pred_mlp, tn_mlp_predict = mlp_predict(clf_mlp, X_test_pca)

    # Accuracy Scores for SVM
    as_svm = accuracy_score(y_test, y_pred_svm)
    as_mlp = accuracy_score(y_test, y_pred_mlp)

    return [as_svm, as_mlp, tn_svm_train, tn_mlp_train, tn_svm_predict, tn_mlp_predict, trainset_size, testset_size]


as_svm_arr = []
as_mlp_arr = []
tn_svm_train_arr = []
tn_mlp_train_arr = []
tn_svm_predict_arr = []
tn_mlp_predict_arr = []
trainset_size_arr = []
testsetsize_arr = []
mnfpp = []

for i in range(60, 91, 10):
    print(i)
    [as_svm, as_mlp, tn_svm_train, tn_mlp_train, tn_svm_predict, tn_mlp_predict, trainset_size, testset_size] = comparison(i)
    print(as_mlp)
    as_svm_arr.append(as_svm)
    as_mlp_arr.append(as_mlp)

    tn_svm_train_arr.append(tn_svm_train)
    tn_mlp_train_arr.append(tn_mlp_train)

    tn_svm_predict_arr.append(tn_svm_predict)
    tn_mlp_predict_arr.append(tn_mlp_predict)

    trainset_size_arr.append(trainset_size)
    testsetsize_arr.append(testset_size)

    mnfpp.append(i)

plt.plot(mnfpp, as_svm_arr, label='Accuracy for SVM')
plt.plot(mnfpp, as_mlp_arr, label='Accuracy for MLP')
plt.xlabel('Minimum Faces per person')
plt.ylabel('Accuracy Observed')
plt.legend()
plt.show()

plt.clf()
plt.plot(trainset_size_arr, tn_svm_train_arr, label='Training Time for SVM')
plt.plot(trainset_size_arr, tn_mlp_train_arr, label='Training Time for MLP')
plt.xlabel('Training Dataset Size')
plt.ylabel('Training Time Observed')
plt.legend()
plt.show()

plt.clf()
plt.plot(testsetsize_arr, tn_svm_predict_arr, label='Recognition Time for SVM')
plt.plot(testsetsize_arr, tn_mlp_predict_arr, label='Recognition Time for MLP')
plt.xlabel('Testing Dataset Size')
plt.ylabel('Recognition Time Observed')
plt.legend()
plt.show()

# Conclusion: MLP is faster at recognition task alone and does
#   not really gets effected by the dataset size and hence recognition task alone,
#   MLP is scalable.
#   However, SVM is faster overall with training and recognition combined, and also
#   has a higher accuracy, unless MLP is fine tuned as well, which gets computationally
#   expensive.
