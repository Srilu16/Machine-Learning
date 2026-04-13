from sklearn.neural_network import MLPClassifier
import numpy as np

inputs = np.array([[0,0],
                   [0,1],
                   [1,0],
                   [1,1]])

and_target = np.array([0,0,0,1])
xor_target = np.array([0,1,1,0])

mlp_and = MLPClassifier(hidden_layer_sizes=(2,),
                        activation='logistic',
                        solver='adam',
                        learning_rate_init=0.05,
                        max_iter=1000,
                        random_state=42)

mlp_and.fit(inputs, and_target)

print("\nAND Gate Results:")
for x, y in zip(inputs, and_target):
    pred = mlp_and.predict([x])[0]
    print("Input:", x, "| Predicted:", pred, "| Actual:", y)

mlp_xor = MLPClassifier(hidden_layer_sizes=(2,),
                        activation='logistic',
                        solver='adam',
                        learning_rate_init=0.05,
                        max_iter=1000,
                        random_state=42)

mlp_xor.fit(inputs, xor_target)

print("\nXOR Gate Results:")
for x, y in zip(inputs, xor_target):
    pred = mlp_xor.predict([x])[0]
    print("Input:", x, "| Predicted:", pred, "| Actual:", y)
