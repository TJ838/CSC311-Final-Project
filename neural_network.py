import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch
import matplotlib.pyplot as plt


from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)


def load_data(base_path="./data"):
    """Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        hidden = torch.sigmoid(self.g(inputs))
        out = torch.sigmoid(self.h(hidden))

        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    #

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    for epoch in range(0, num_epoch):
        train_loss = 0.0

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[nan_mask] = output[nan_mask]

            loss = torch.sum((output - target) ** 2.0)
            loss += lamb * model.get_weight_norm()
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print(
            "Epoch: {} \tTraining Cost: {:.6f}\t " "Valid Acc: {}".format(
                epoch, train_loss, valid_acc
            )
        )
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def train_and_plot(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """Train the model while capturing the training loss and validation accuracy,
       then plot them at the end of training.
    """
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    train_losses = []
    valid_accuracies = []

    model.train()

    for epoch in range(num_epoch):
        train_loss = 0.0

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[nan_mask] = output[nan_mask]

            loss = torch.sum((output - target) ** 2.0)
            loss += lamb * model.get_weight_norm()
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        train_losses.append(train_loss / num_student)
        valid_acc = evaluate(model, zero_train_data, valid_data)
        valid_accuracies.append(valid_acc)

        print(
            "Epoch: {} \tTraining Cost: {:.6f}\t " "Valid Acc: {}".format(
                epoch, train_loss, valid_acc
            )
        )

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(num_epoch), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs. Epoch')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(num_epoch), valid_accuracies, label='Validation Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy vs. Epoch')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    k_values = [10, 50, 100, 200, 500]
    best_k = None
    best_val_acc = 0
    best_model = None

    lr = 0.01
    num_epoch = 100
    lamb = 0

    for k in k_values:
        print(f"Training with k = {k}...")
        model = AutoEncoder(num_question=train_matrix.shape[1], k=k)
        train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)

        val_acc = evaluate(model, zero_train_matrix, valid_data)
        print(f"Validation accuracy for k={k}: {val_acc}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_k = k
            best_model = model

    print(f"Best k: {best_k} with validation accuracy: {best_val_acc}")

    train_and_plot(best_model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)

    test_acc = evaluate(best_model, zero_train_matrix, test_data)
    print(f"Test accuracy: {test_acc}")

    lamb_values = [0.001, 0.01, 0.1, 1]
    best_lamb = None
    best_lamb_acc = 0

    for lamb in lamb_values:
        print(f"Training with lambda = {lamb}...")
        model = AutoEncoder(num_question=train_matrix.shape[1], k=best_k)
        train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)

        val_acc = evaluate(model, zero_train_matrix, valid_data)
        print(f"Validation accuracy for lambda={lamb}: {val_acc}")

        if val_acc > best_lamb_acc:
            best_lamb_acc = val_acc
            best_lamb = lamb

    print(f"Best lambda: {best_lamb} with validation accuracy: {best_lamb_acc}")

    final_model = AutoEncoder(num_question=train_matrix.shape[1], k=best_k)
    train(final_model, lr, best_lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)

    final_test_acc = evaluate(final_model, zero_train_matrix, test_data)
    print(f"Final Test accuracy with best k={best_k} and best lambda={best_lamb}: {final_test_acc}")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
