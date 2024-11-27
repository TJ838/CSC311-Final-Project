import numpy as np
import matplotlib.pyplot as plt
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)

def matches_divided_by_total(u, v):
    valid = ~np.isnan(u) & ~np.isnan(v)
    if np.sum(valid) == 0:
        return 0.0
    
    similarity = np.sum(u[valid] == v[valid]) / np.sum(valid)
    return similarity

def pairwise_distance(matrix):
    n_samples = matrix.shape[0]
    similarities = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            sim = matches_divided_by_total(matrix[i], matrix[j])
            similarities[i, j] = sim
            similarities[j, i] = sim

    return 1 - similarities

def knn_impute_with_smc(matrix, valid_data, k):
    distances = pairwise_distance(matrix)

    # Initialize the imputed matrix as a copy of the original matrix
    imputed_matrix = np.copy(matrix)

    # Iterate through each row/sample
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isnan(matrix[i, j]):
                # Get k nearest neighbors
                neighbors_idx = np.argsort(distances[i, :])[:k]

                # Collect valid values and their weights from neighbors
                neighbor_values = matrix[neighbors_idx, j]
                valid_mask = ~np.isnan(neighbor_values)
                valid_values = neighbor_values[valid_mask]
                # Use 1 - distance as weights
                weights = 1 - distances[i, neighbors_idx][valid_mask]

                # If there are valid values, impute using weighted mean
                if valid_values.size > 0:
                    if np.sum(weights) > 0:
                        # Check if the sum of weights is non-zero
                        imputed_matrix[i, j] = np.average(valid_values, weights=weights)
                    else:
                        # If weights sum to zero, avoid division by zero
                        imputed_matrix[i, j] = np.mean(valid_values)

    # Evaluate accuracy using the imputed matrix
    acc = sparse_matrix_evaluate(valid_data, imputed_matrix)
    print(f"Validation Accuracy (k={k}): {acc}")
    return acc


def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    k_values = [60, 80, 100, 120, 140, 180]
    user_accuracies = []

    for k in k_values:
        acc = knn_impute_with_smc(sparse_matrix, val_data, k)
        user_accuracies.append(acc)

    k_star_user = k_values[np.argmax(user_accuracies)]

    print("\nBest k:", k_star_user)

    print("\nTest Accuracy on k*:")
    knn_impute_with_smc(sparse_matrix, test_data, k_star_user)

if __name__ == "__main__":
    main()