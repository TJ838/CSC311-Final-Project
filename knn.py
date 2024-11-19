import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def knn_impute_by_user(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html 
    for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy (User-based, k={}): {}".format(k, acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    nbrs = KNNImputer(n_neighbors=k)
    # sparse_matrix is row = user, column = question, so
    # transpose the matrix for item/question-based similarity.
    mat = nbrs.fit_transform(matrix.T).T
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy (Item-based, k={}): {}".format(k, acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################

    k_values = [1, 6, 11, 16, 21, 26]
    user_accuracies = []
    item_accuracies = []

    # user based
    # part (a)
    for k in k_values:
        acc = knn_impute_by_user(sparse_matrix, val_data, k)
        user_accuracies.append(acc)

    print("\n")
    # item based
    # part (c)
    for k in k_values:
        acc = knn_impute_by_item(sparse_matrix, val_data, k)
        item_accuracies.append(acc)

    # plot accuracy
    # part (a)/(c)
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, user_accuracies, label="User-based", marker="o")
    plt.plot(k_values, item_accuracies, label="Item-based", marker="o")
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs. k")
    plt.legend()
    plt.grid()
    plt.show()

    # part (b)
    k_star_user = k_values[np.argmax(user_accuracies)]
    k_star_item = k_values[np.argmax(item_accuracies)]

    # part (b)
    print("\nBest k (User-based):", k_star_user)
    print("Best k (Item-based):", k_star_item)

    # test accuracy on test data for the best k*
    print("\nTest Accuracy on k*:")
    knn_impute_by_user(sparse_matrix, test_data, k_star_user)
    knn_impute_by_item(sparse_matrix, test_data, k_star_item)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
