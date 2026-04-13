import math

def bipolar_step(x):
    return 1 if x >= 0 else -1

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def relu(x):
    return max(0, x)

def train(activation, targets, threshold, max_epochs=1000):
    inputs = [(0,0), (0,1), (1,0), (1,1)]

    w0 = 10
    w1 = 0.2
    w2 = -0.75
    alpha = 0.05

    epoch = 0

    while epoch < max_epochs:
        total_error = 0

        for i in range(len(inputs)):
            x1, x2 = inputs[i]
            target = targets[i]

            net = w0*1 + w1*x1 + w2*x2
            output = activation(net)

            error = target - output
            total_error += error**2

            w0 = w0 + alpha * error * 1
            w1 = w1 + alpha * error * x1
            w2 = w2 + alpha * error * x2

        epoch += 1

        if total_error <= threshold:
            break

    return epoch

epochs_bipolar = train(bipolar_step, [-1, -1, -1, 1], 0)
epochs_sigmoid = train(sigmoid, [0, 0, 0, 1], 0.01)
epochs_relu = train(relu, [0, 0, 0, 1], 0)

print("Iterations to Converge:")
print("Bi-Polar Step:", epochs_bipolar)
print("Sigmoid:", epochs_sigmoid)
print("ReLU:", epochs_relu)
