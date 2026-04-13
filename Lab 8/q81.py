import math

def summation_unit(inputs, weights, bias):
    return sum(i * w for i, w in zip(inputs, weights)) + bias

def step_activation(x):
    return 1 if x >= 0 else 0

def bipolar_step(x):
    return 1 if x >= 0 else -1

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def tanh_activation(x):
    return math.tanh(x)

def relu(x):
    return max(0, x)

def leaky_relu(x):
    alpha = 0.01
    return x if x > 0 else alpha * x

def simple_error(target, output):
    return target - output

def mse(target, output):
    return (target - output) ** 2

inputs = [1, 2]
weights = [0.5, 0.3]
bias = 0.1
target = 1

net = summation_unit(inputs, weights, bias)

out_step = step_activation(net)
out_bipolar = bipolar_step(net)
out_sigmoid = sigmoid(net)
out_tanh = tanh_activation(net)
out_relu = relu(net)
out_leaky = leaky_relu(net)

err = simple_error(target, out_sigmoid)
mse_val = mse(target, out_sigmoid)

print("Summation (Net Input):", net)
print("Step Activation:", out_step)
print("Bipolar Step Activation:", out_bipolar)
print("Sigmoid Activation:", out_sigmoid)
print("TanH Activation:", out_tanh)
print("ReLU Activation:", out_relu)
print("Leaky ReLU Activation:", out_leaky)
print("Error (Target - Output):", err)
print("Mean Squared Error:", mse_val)
