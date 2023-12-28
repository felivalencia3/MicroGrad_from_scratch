import random
from engine import Value
from util import draw_dot


class Neuron:
    def __init__(self, n_in):
        # randomly start weights and bias
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(random.uniform(-1, 1))

    def parameters(self):
        return self.w + [self.b]

    def __call__(self, x):
        # w * x (vectors) + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out


class Layer:
    def __init__(self, n_in, n_out):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]



class MultiLayerPerceptron:
    def __init__(self, n_in, n_outs: list):
        sz = [n_in] + n_outs
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(n_outs))]

    def __call__(self, x) -> list[Value]:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


x = [2.0, 3.0, -1.0]
n = MultiLayerPerceptron(3, [4, 4, 1])
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

ypred = []
# gradient descent
for i in range(100):
    # forward pass
    ypred = [n(x)[0] for x in xs]
    loss = Value(0)
    for y_ground_truth, y_out in zip(ys, ypred):
        loss += (y_out - y_ground_truth) ** 2

    # backward pass + clear grads
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    # update
    for p in n.parameters():
        # move towards negative gradient
        p.data += -0.05 * p.grad

    print(i, loss.data)

print(ypred)
