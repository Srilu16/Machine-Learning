import math
import numpy as np

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

data = [
    ([20,6,2,386],1),
    ([16,3,6,289],1),
    ([27,6,2,393],1),
    ([19,1,2,110],0),
    ([24,4,2,280],1),
    ([22,1,5,167],0),
    ([15,4,2,271],1),
    ([18,4,2,274],1),
    ([21,1,4,148],0),
    ([16,2,4,198],0)
]

w = [0.4, -0.2, 0.3, 0.1]
b = 0.1
lr = 0.05

for _ in range(1000):
    error_sum = 0

    for x, y in data:

        x_n = [x[0]/30, x[1]/10, x[2]/10, x[3]/400]

        net = sum(w[j]*x_n[j] for j in range(4)) + b
        o = sigmoid(net)

        e = y - o
        error_sum += e**2

        d = e * o * (1 - o)

        for j in range(4):
            w[j] += lr * d * x_n[j]

        b += lr * d

    if error_sum < 0.001:
        break

print("A6 Weights:", w)
print("Bias:", b)

print("\nA6 Predictions:")
for idx, (x, y) in enumerate(data, 1):

    x_n = [x[0]/30, x[1]/10, x[2]/10, x[3]/400]

    net = sum(w[j]*x_n[j] for j in range(4)) + b
    o = sigmoid(net)

    label = 1 if o >= 0.5 else 0
    print("Customer", idx, "| Pred:", label, "| Actual:", y)

X = np.array([
    [20,6,2,386],
    [16,3,6,289],
    [27,6,2,393],
    [19,1,2,110],
    [24,4,2,280],
    [22,1,5,167],
    [15,4,2,271],
    [18,4,2,274],
    [21,1,4,148],
    [16,2,4,198]
], float)

y = np.array([1,1,1,0,1,0,1,1,0,0], float)

X[:,0]/=30
X[:,1]/=10
X[:,2]/=10
X[:,3]/=400

Xb = np.c_[X, np.ones(len(X))]

W = np.linalg.pinv(Xb) @ y

pred = (Xb @ W >= 0.5).astype(int)

print("\nPseudo-Inverse Weights:", W)

print("\nA7 Predictions:")
for i in range(len(X)):
    print("Customer", i+1, "| Pred:", pred[i], "| Actual:", int(y[i]))
