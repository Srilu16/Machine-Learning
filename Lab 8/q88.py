import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(output):
    return output * (1 - output)

inputs = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])

targets = np.array([[0], [0], [0], [1]])

np.random.seed(42)

V = np.random.uniform(-0.5, 0.5, (2, 2))
W = np.random.uniform(-0.5, 0.5, (2, 1))

b_hidden = np.zeros((1, 2))
b_output = np.zeros((1, 1))

learning_rate = 0.05
max_iterations = 1000
convergence_error = 0.002

for iteration in range(1, max_iterations + 1):
    total_error = 0

    for i in range(len(inputs)):
        x = inputs[i].reshape(1, 2)
        t = targets[i].reshape(1, 1)

        net_h = np.dot(x, V) + b_hidden
        H = sigmoid(net_h)

        net_o = np.dot(H, W) + b_output
        O = sigmoid(net_o)

        error = 0.5 * (t - O) ** 2
        total_error += error[0, 0]

        delta_o = (t - O) * sigmoid_derivative(O)
        delta_h = sigmoid_derivative(H) * np.dot(delta_o, W.T)

        W += learning_rate * H.T.dot(delta_o)
        b_output += learning_rate * delta_o

        V += learning_rate * x.T.dot(delta_h)
        b_hidden += learning_rate * delta_h

    if total_error <= convergence_error:
        break

for i in range(len(inputs)):
    x = inputs[i].reshape(1, 2)
    H = sigmoid(np.dot(x, V) + b_hidden)
    O = sigmoid(np.dot(H, W) + b_output)

    pred = 1 if O[0, 0] >= 0.5 else 0

    print("Input:", inputs[i], "| Predicted:", pred, "| Actual:", targets[i][0])
