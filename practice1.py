import numpy as np
import random
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

sizes=[2,3,1]
num_layers = len(sizes)
biases = [np.random.randn(y, 1) for y in sizes[1:]]
print(biases)
weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

print(weights)
#weights = [print(y,x) for x, y in zip(sizes[:-1], sizes[1:])]
for b, w in zip(biases,weights):
            a = sigmoid(np.dot(w, a)+b)
            print(a)