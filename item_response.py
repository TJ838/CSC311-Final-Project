from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    log_lklihood = 0.0

    for user_id, question_id, is_correct in zip(data["user_id"], data["question_id"], data["is_correct"]):
        diff = theta[user_id] - beta[question_id]

        if is_correct:
            log_lklihood += diff - np.log(1 + np.exp(diff))
        else:
            log_lklihood += -np.log(1 + np.exp(diff))

    return log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    theta_grad = np.zeros_like(theta)
    beta_grad = np.zeros_like(beta)

    for user_id, question_id, is_correct in zip(data["user_id"], data["question_id"], data["is_correct"]):
        diff = theta[user_id] - beta[question_id]
        theta_grad[user_id] += is_correct - sigmoid(diff)
        beta_grad[question_id] += sigmoid(diff) - is_correct

    theta += lr * theta_grad
    beta += lr * beta_grad

    return theta, beta


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst, lld_train_lst, lld_val_lst)
    """
    np.random.seed(88)
    theta = np.random.rand(len(np.unique(data["user_id"])))
    beta = np.random.rand(len(np.unique(data["question_id"])))

    val_acc_lst = []
    lld_train_lst = []
    lld_val_lst = []

    for i in range(iterations):
        neg_lld_train = neg_log_likelihood(data, theta=theta, beta=beta)
        neg_lld_val = neg_log_likelihood(val_data, theta=theta, beta=beta)
        
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        lld_train_lst.append(neg_lld_train)
        lld_val_lst.append(neg_lld_val)
        print("NLLK: {} \t Score: {}".format(neg_lld_train, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    return theta, beta, val_acc_lst, lld_train_lst, lld_val_lst


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    lr = 0.005
    iterations = 100

    theta, beta, val_acc_lst, lld_train_lst, lld_val_lst = irt(train_data, val_data, lr, iterations)

    test_acc = evaluate(test_data, theta, beta)
    
    print("Test Accuracy: {}".format(test_acc))
    print("Validation Accuracy: {}".format(val_acc_lst[-1]))

    plt.plot(range(1, len(val_acc_lst) + 1), val_acc_lst, label="Validation Accuracy", color="red")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Training Curve: Validation Accuracy vs Iterations")
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(range(1, len(lld_train_lst) + 1), lld_train_lst, label="Training Log-Likelihood", color="blue")
    plt.plot(range(1, len(lld_val_lst) + 1), lld_val_lst, label="Validation Log-Likelihood", color="orange")
    plt.xlabel("Iterations")
    plt.ylabel("Log-Likelihood")
    plt.title("Training and Validation Log-Likelihoods vs Iterations")
    plt.legend()
    plt.grid()
    plt.show()

    # part d
    questions = [8, 88, 188]
    theta_range = np.linspace(-4, 4, 100)

    plt.figure(figsize=(8, 6))

    for q in questions:
        prob = sigmoid(theta_range - beta[q])
        # print("difficulty of question {} is {}".format(q, beta[q]))
        plt.plot(theta_range, prob, label=f"Question {q}")
    
    plt.xlabel("Theta (Ability)")
    plt.ylabel("Probability of Correct Response")
    plt.title("Probability of Correct Response vs Theta")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
