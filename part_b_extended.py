import numpy as np
from sklearn.impute import KNNImputer
import time
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

def knn_impute_by_user(matrix, valid_data, k):
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
    return acc


def increase_sparsity(matrix, additional_sparsity_level):
    non_nan_indices = np.where(~np.isnan(matrix))
    non_nan_count = len(non_nan_indices[0])

    # Determine the number of entries to make NaN
    entries_to_nan = int(additional_sparsity_level * non_nan_count)

    # Randomly select indices of non-NaN entries to make NaN
    random_indices = np.random.choice(non_nan_count, entries_to_nan, replace=False)

    sparse_matrix = matrix.copy()

    # Set selected entries to NaN
    sparse_matrix[
        non_nan_indices[0][random_indices], non_nan_indices[1][random_indices]
    ] = np.nan

    return sparse_matrix


def calculate_sparsity(matrix):
    total_entries = np.prod(matrix.shape)
    missing_entries = np.sum(np.isnan(matrix))
    sparsity = missing_entries / total_entries
    return sparsity


def main():
    # Load data
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    # Calculate original sparsity
    original_sparsity = calculate_sparsity(sparse_matrix)
    print(f"Original Sparsity of Dataset: {original_sparsity * 100:.2f}%")

    sparsity_levels = [0.009, 0.024, 0.039, 0.055]
    k_values = [10, 40, 70, 100, 130, 160, 190]
    results = {}

    for sparsity in sparsity_levels:
        print(f"\nTesting with additional sparsity: {sparsity}")
        sparse_matrix_mod = increase_sparsity(sparse_matrix, sparsity)

        accuracies = []
        times = []
        for k in k_values:
            start_time = time.time()

            acc = knn_impute_by_user(sparse_matrix_mod, val_data, k)

            end_time = time.time()
            elapsed_time = end_time - start_time

            accuracies.append(acc)
            times.append(elapsed_time)

            print(f"Sparsity: {94.1 + sparsity * 100:.1f}%, k: {k},"
                  f" Validation Accuracy: {acc:.4f}, "
                  f"Time Taken: {elapsed_time:.4f} seconds")

        # Record results
        results[sparsity] = {'accuracies': accuracies, 'times': times}

    # Plot results
    plt.figure(figsize=(12, 8))
    for sparsity, data in results.items():
        accuracies = data['accuracies']
        plt.plot(k_values, accuracies, label=f""
                                             f"Sparsity "
                                             f"{94.1 + sparsity * 100:.1f}"
                                             f"%", marker="o")

    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs. k for Different Sparsity Levels")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
