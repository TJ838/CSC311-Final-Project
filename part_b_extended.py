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


# Function to perform KNN imputation
def knn_impute_by_user(matrix, valid_data, k):
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    return acc


# Function to increase sparsity by making random values NaN
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


# Function to calculate sparsity level of the matrix
def calculate_sparsity(matrix):
    total_entries = np.prod(matrix.shape)
    missing_entries = np.sum(np.isnan(matrix))
    sparsity = missing_entries / total_entries
    return sparsity


# Main function
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

    times_results = {}

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

            print(f"Sparsity: {94.1 + sparsity * 100:.1f}%, k: "
                  f"{k}, Validation Accuracy: {acc:.4f}, "
                  f"Time Taken: {elapsed_time:.4f} seconds")

        results[sparsity] = accuracies
        times_results[sparsity] = times

    # Plot results
    plt.figure(figsize=(12, 8))

    for sparsity, accuracies in results.items():
        plt.plot(k_values, accuracies, label=f""
                                             f"Sparsity "
                                             f"{94.1 + sparsity * 100:.1f}"
                                             f"%", marker="o")

    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs. "
              "k for Different Sparsity Levels (User-based kNN)")
    plt.legend()
    plt.grid()
    plt.show()

    for sparsity in sparsity_levels:
        print(f"\nRunning times for Sparsity {94.1 + sparsity * 100:.1f}%:")
        for k, time_taken in zip(k_values, times_results[sparsity]):
            print(f"k = {k}: {time_taken:.4f} seconds")


if __name__ == "__main__":
    main()
