from torch.autograd import Variable
from sklearn.impute import KNNImputer

import item_response as ir
import knn
import neural_network as nn
import numpy as np
import torch

from utils import (
    load_train_sparse,
    load_valid_csv,
    load_public_test_csv,
)

def bootstrap_sparse_matrix(sparse_matrix):
    # Get total number of rows
    num_rows = sparse_matrix.shape[0]
    
    # Sample num_rows number of row indices from [1,...,num_rows] with replacement
    sampled_indices = np.random.choice(num_rows, num_rows, replace=True)
    
    # Create the bootstrapped sparse matrix using sampled rows
    bootstrapped_matrix = sparse_matrix[sampled_indices, :]
    
    return bootstrapped_matrix

def convert_sparse_matrix_to_dict(sparse_matrix):
    # Initialize the data.
    data = {"user_id": [], "question_id": [], "is_correct": []}

    # Loop through each entry and add to dict if has a value recorded
    for row in range(sparse_matrix.shape[0]):
        for col in range(sparse_matrix.shape[1]):
            if (not np.isnan(sparse_matrix[row][col])):
                data["question_id"].append(col)
                data["user_id"].append(row)
                data["is_correct"].append(sparse_matrix[row][col])

    return data

def knn_model(sparse_matrix, val_data, test_data):
    train_matrix = bootstrap_sparse_matrix(sparse_matrix)

    k_values = [1, 6, 11, 16, 21, 26]
    user_accuracies = []

    for k in k_values:
        acc = knn.knn_impute_by_user(train_matrix, val_data, k)
        user_accuracies.append(acc)

    k_star_user = k_values[np.argmax(user_accuracies)]

    val_pred = knn_get_predictions(train_matrix, val_data, k_star_user)
    test_pred = knn_get_predictions(train_matrix, test_data, k_star_user)

    return val_pred, test_pred

def knn_get_predictions(matrix, data, k):
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    pred = []
    for i, u in enumerate(data["user_id"]):
        q = data["question_id"][i]
        pred.append(mat[u, q])

    return np.array(pred)

def item_response_model(sparse_matrix, val_data, test_data):
    train_matrix = bootstrap_sparse_matrix(sparse_matrix)
    train_data = convert_sparse_matrix_to_dict(train_matrix)

    lr = 0.005
    iterations = 100

    theta, beta, _, _, _ = ir.irt(train_data, val_data, lr, iterations)

    val_pred = item_response_get_predictions(val_data, theta, beta)
    test_pred = item_response_get_predictions(test_data, theta, beta)

    return val_pred, test_pred

def item_response_get_predictions(data, theta, beta):
    pred = []
    for i, u in enumerate(data["user_id"]):
        q = data["question_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = ir.sigmoid(x)
        pred.append(p_a)
    return np.array(pred)

def load_data_neural_network(sparse_matrix):
    train_matrix = bootstrap_sparse_matrix(sparse_matrix)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix

def neural_network_model(sparse_matrix, valid_data, test_data):
    zero_train_matrix, train_matrix = load_data_neural_network(sparse_matrix)

    k_values = [10, 50, 100, 200, 500]
    best_k = None
    best_val_acc = 0

    lr = 0.01
    num_epoch = 50
    lamb = 0

    for k in k_values:
        model = nn.AutoEncoder(num_question=train_matrix.shape[1], k=k)
        nn.train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)

        val_acc = nn.evaluate(model, zero_train_matrix, valid_data)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_k = k

    lamb_values = [0.001, 0.01, 0.1, 1]
    best_lamb = None
    best_lamb_acc = 0

    for lamb in lamb_values:
        model = nn.AutoEncoder(num_question=train_matrix.shape[1], k=best_k)
        nn.train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)

        val_acc = nn.evaluate(model, zero_train_matrix, valid_data)

        if val_acc > best_lamb_acc:
            best_lamb_acc = val_acc
            best_lamb = lamb

    final_model = nn.AutoEncoder(num_question=train_matrix.shape[1], k=best_k)
    nn.train(final_model, lr, best_lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)

    val_pred = neural_network_get_prediction(final_model, zero_train_matrix, valid_data)
    test_pred = neural_network_get_prediction(final_model, zero_train_matrix, test_data)

    return val_pred, test_pred

def neural_network_get_prediction(model, train_data, data):
    # Tell PyTorch you are evaluating the model.
    model.eval()

    pred = []

    for i, u in enumerate(data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][data["question_id"][i]].item()
        pred.append(guess)
    return np.array(pred)

def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    ir_val_pred, ir_test_pred = item_response_model(sparse_matrix, val_data, test_data)
    nn_val_pred, nn_test_pred = neural_network_model(sparse_matrix, val_data, test_data)
    knn_val_pred, knn_test_pred = knn_model(sparse_matrix, val_data, test_data)

    val_pred = ((ir_val_pred + nn_val_pred + knn_val_pred) / 3) >= 0.5
    test_pred = ((ir_test_pred + nn_test_pred + knn_test_pred) / 3) >= 0.5

    val_acc = np.sum((val_data["is_correct"] == np.array(val_pred))) / len(val_data["is_correct"])
    test_acc = np.sum((test_data["is_correct"] == np.array(test_pred))) / len(test_data["is_correct"])

    # debugging/verification
    print("IRT first 10 predictions:", ir_val_pred[:10])
    print("NN first 10 predictions:", nn_val_pred[:10])
    print("KNN first 10 predictions:", knn_val_pred[:10])
    print("Final first 10 predictions:", val_pred[:10])

    ir_val_binary = ir_val_pred >= 0.5
    nn_val_binary = nn_val_pred >= 0.5
    knn_val_binary = knn_val_pred >= 0.5

    ir_val_acc = np.sum((val_data["is_correct"] == np.array(ir_val_binary))) / len(val_data["is_correct"])
    nn_val_acc = np.sum((val_data["is_correct"] == np.array(nn_val_binary))) / len(val_data["is_correct"])
    knn_val_acc = np.sum((val_data["is_correct"] == np.array(knn_val_binary))) / len(val_data["is_correct"])

    print(f"IRT Validation Accuracy: {ir_val_acc}")
    print(f"NN Validation Accuracy: {nn_val_acc}")
    print(f"KNN Validation Accuracy: {knn_val_acc}")

    ir_test_binary = ir_test_pred >= 0.5
    nn_test_binary = nn_test_pred >= 0.5
    knn_test_binary = knn_test_pred >= 0.5

    ir_test_acc = np.sum((test_data["is_correct"] == np.array(ir_test_binary))) / len(test_data["is_correct"])
    nn_test_acc = np.sum((test_data["is_correct"] == np.array(nn_test_binary))) / len(test_data["is_correct"])
    knn_test_acc = np.sum((test_data["is_correct"] == np.array(knn_test_binary))) / len(test_data["is_correct"])

    print(f"IRT Test Accuracy: {ir_test_acc}")
    print(f"NN Test Accuracy: {nn_test_acc}")
    print(f"KNN Test Accuracy: {knn_test_acc}")

    print(f"Final validation accuracy: {val_acc}, final test accuracy: {test_acc}")

if __name__ == "__main__":
    main()