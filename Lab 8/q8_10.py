import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(output):
    return output * (1 - output)

inputs = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])

targets_and = np.array([[1, 0],
                        [1, 0],
                        [1, 0],
                        [0, 1]])

np.random.seed(42)
V = np.random.uniform(-0.5, 0.5, (2, 2))
W = np.random.uniform(-0.5, 0.5, (2, 2))
b_hidden = np.zeros((1, 2))
b_output = np.zeros((1, 2))

learning_rate = 0.05
max_iterations = 1000
convergence_error = 0.002

for _ in range(max_iterations):
    total_error = 0

    for i in range(len(inputs)):
        x = inputs[i].reshape(1, 2)
        t = targets_and[i].reshape(1, 2)

        H = sigmoid(np.dot(x, V) + b_hidden)
        O = sigmoid(np.dot(H, W) + b_output)

        total_error += 0.5 * np.sum((t - O) ** 2)

        delta_o = (t - O) * sigmoid_derivative(O)
        delta_h = sigmoid_derivative(H) * np.dot(delta_o, W.T)

        W += learning_rate * np.dot(H.T, delta_o)
        b_output += learning_rate * delta_o
        V += learning_rate * np.dot(x.T, delta_h)
        b_hidden += learning_rate * delta_h

    if total_error <= convergence_error:
        break

print("\nAND Gate Predictions:\n")
for i in range(len(inputs)):
    x = inputs[i].reshape(1, 2)
    H = sigmoid(np.dot(x, V) + b_hidden)
    O = sigmoid(np.dot(H, W) + b_output)

    pred = [1 if O[0, j] >= 0.5 else 0 for j in range(2)]
    print("Input:", inputs[i], "| Predicted:", pred, "| Actual:", targets_and[i].tolist())


targets_or = np.array([[1, 0],
                       [0, 1],
                       [0, 1],
                       [0, 1]])

np.random.seed(42)
V = np.random.uniform(-0.5, 0.5, (2, 2))
W = np.random.uniform(-0.5, 0.5, (2, 2))
b_hidden = np.zeros((1, 2))
b_output = np.zeros((1, 2))

for _ in range(max_iterations):
    total_error = 0

    for i in range(len(inputs)):
        x = inputs[i].reshape(1, 2)
        t = targets_or[i].reshape(1, 2)

        H = sigmoid(np.dot(x, V) + b_hidden)
        O = sigmoid(np.dot(H, W) + b_output)

        total_error += 0.5 * np.sum((t - O) ** 2)

        delta_o = (t - O) * sigmoid_derivative(O)
        delta_h = sigmoid_derivative(H) * np.dot(delta_o, W.T)

        W += learning_rate * np.dot(H.T, delta_o)
        b_output += learning_rate * delta_o
        V += learning_rate * np.dot(x.T, delta_h)
        b_hidden += learning_rate * delta_h

    if total_error <= convergence_error:
        break

print("\nOR Gate Predictions:\n")
for i in range(len(inputs)):
    x = inputs[i].reshape(1, 2)
    H = sigmoid(np.dot(x, V) + b_hidden)
    O = sigmoid(np.dot(H, W) + b_output)

    pred = [1 if O[0, j] >= 0.5 else 0 for j in range(2)]
    print("Input:", inputs[i], "| Predicted:", pred, "| Actual:", targets_or[i].tolist())
