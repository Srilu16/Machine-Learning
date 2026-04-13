import matplotlib.pyplot as plt

def activation(net):
    return 1 if net >= 0 else 0

def perceptron_train():
    inputs = [(0,0), (0,1), (1,0), (1,1)]
    targets = [0, 0, 0, 1]

    w0 = 10
    w1 = 0.2
    w2 = -0.75
    alpha = 0.05

    errors = []
    epoch = 0

    while True:
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

        errors.append(total_error)
        epoch += 1

        if total_error == 0:
            break

    return w0, w1, w2, epoch, errors

w0, w1, w2, epochs, errors = perceptron_train()

print("Final Weights:")
print("w0 =", w0)
print("w1 =", w1)
print("w2 =", w2)

print("\nEpochs for convergence:", epochs)

print("\nAND Gate Outputs after Training:")
inputs = [(0,0), (0,1), (1,0), (1,1)]
for x1, x2 in inputs:
    net = w0*1 + w1*x1 + w2*x2
    print(f"Input ({x1},{x2}) -> Output:", activation(net))

plt.plot(range(1, epochs+1), errors)
plt.xlabel("Epochs")
plt.ylabel("Sum Squared Error")
plt.title("Epoch vs Error")
plt.show()
