import matplotlib.pyplot as plt

def step(x):
    return 1 if x >= 0 else 0

def bipolar(x):
    return 1 if x >= 0 else -1

def sigmoid(x):
    return 1 / (1 + (2.71828 ** -x))

def relu(x):
    return max(0, x)

def train(activation, alpha, targets, threshold):
    inputs = [(0,0), (0,1), (1,0), (1,1)]

    w0, w1, w2 = 10, 0.2, -0.75

    epoch = 0
    errors = []

    while epoch < 1000:
        total_error = 0

        for i in range(len(inputs)):
            x1, x2 = inputs[i]
            target = targets[i]

            net = w0*1 + w1*x1 + w2*x2
            output = activation(net)

            error = target - output
            total_error += error**2

            w0 += alpha * error * 1
            w1 += alpha * error * x1
            w2 += alpha * error * x2

        errors.append(total_error)
        epoch += 1

        if total_error <= threshold:
            break

    return epoch, errors


targets = [0, 1, 1, 0]

ep_step, err_step = train(step, 0.05, targets, 0.002)
ep_bipolar, err_bipolar = train(bipolar, 0.05, targets, 0.002)
ep_sigmoid, err_sigmoid = train(sigmoid, 0.05, targets, 0.002)
ep_relu, err_relu = train(relu, 0.05, targets, 0.002)

print("Epochs to Converge:")
print("Step:", ep_step)
print("Bi-Polar:", ep_bipolar)
print("Sigmoid:", ep_sigmoid)
print("ReLU:", ep_relu)


learning_rates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
epochs_lr = []

for lr in learning_rates:
    ep, _ = train(step, lr, targets, 0.002)
    epochs_lr.append(ep)

plt.plot(learning_rates, epochs_lr, marker='o')
plt.xlabel("Learning Rate")
plt.ylabel("Epochs to Converge")
plt.title("XOR: Learning Rate vs Iterations")
plt.show()
