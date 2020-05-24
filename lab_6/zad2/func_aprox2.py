import numpy as np
from matplotlib import pyplot as plt


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2


def sigmoid(x):
    return 1.0/(1+np.exp(-x))


def sigmoid_derivative(x):
    return x*(1.0-x)


def ReLU(x):
    b = x.copy()
    b[b < 0.0] = 0.0
    return b


def ReLU_derivative(x):
    return 1*(x >= 0)


class NeuralNetwork:
    def __init__(self, x, y, hidden_layer_size=50):
        self.input = x
        self.weights1 = np.random.rand(hidden_layer_size, 2)
        self.weights2 = np.random.rand(hidden_layer_size, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)
        self.eta = 0.01

    def feedforward(self):
        self.layer1 = tanh(np.dot(self.weights1, self.input.T))
        self.output = tanh(np.dot(self.layer1.T, self.weights2))

    def backpropagation(self):
        delta2 = (self.y - self.output) * \
            tanh_derivative(self.output)
        d_weights2 = self.eta * np.dot(self.layer1, delta2)

        delta1 = tanh_derivative(self.layer1) * \
            np.dot(self.weights2, delta2.T)
        d_weights1 = self.eta * np.dot(delta1, self.input)

        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def prediction(self, x):
        layer_pred = tanh(np.dot(self.weights1, x.T))
        return tanh(np.dot(layer_pred.T, self.weights2))


if __name__ == "__main__":
    x = np.reshape(np.linspace(0, 2, 21), (21, 1))
    y = np.sin((3*np.pi/2) * x)
    x_s = np.ones((21, 1))
    x = np.concatenate((x, x_s), axis=1)  # BIAS
    nn = NeuralNetwork(x, y)

    x_t = np.reshape(np.linspace(0, 2, 161), (161, 1))
    y_t = np.sin((3*np.pi/2) * x_t)
    x_s = np.ones((161, 1))
    x_test = np.concatenate((x_t, x_s), axis=1)  # BIAS
    while True:
        for i in range(50000):
            nn.feedforward()
            nn.backpropagation()
        pred = nn.prediction(x_test)
        plt.plot(x_t, pred, 'ro', markersize=2, label="prediction")
        plt.plot(x_t, y_t, 'bo', markersize=5, label="valid")
        plt.show()
        print(y)
        print(nn.output)
        print(nn.weights1)
        print(nn.weights2)
