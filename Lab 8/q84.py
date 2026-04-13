import matplotlib.pyplot as plt

def step(x):
    return 1 if x >= 0 else 0

def train(alpha):
    inputs = [(0,0), (0,1), (1,0), (1,1)]
    targets = [0, 0, 0, 1]

    w0 = 10
    w1 = 0.2
    w2 = -0.75

    epoch = 0

    while epoch < 1000:
        total_error = 0

        for i in range(len(inputs)):
            x1, x2 = inputs[i]
            target = targets[i]

            net = w0*1 + w1*x1 + w2*x2
            output = step(net)

            error = target - output
            total_error += error**2

            w0 = w0 + alpha * error * 1
            w1 = w1 + alpha * error * x1
            w2 = w2 + alpha * error * x2

        epoch += 1

        if total_error == 0:
            break

    return epoch

learning_rates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
epochs_list = []

for lr in learning_rates:
    epochs = train(lr)
    epochs_list.append(epochs)

print("Learning Rate vs Epochs:")
for i in range(len(learning_rates)):
    print(learning_rates[i], "->", epochs_list[i])

plt.plot(learning_rates, epochs_list, marker='o')
plt.xlabel("Learning Rate")
plt.ylabel("Epochs to Converge")
plt.title("Learning Rate vs Iterations")
plt.show()
