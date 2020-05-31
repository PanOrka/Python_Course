import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1.0 - x**2


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
    def __init__(self, x, y, hidden_layer_size=100):
        self.input = x
        self.weights1 = np.random.rand(hidden_layer_size, 2)
        self.weights2 = np.random.rand(hidden_layer_size, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)
        self.eta = 0.000001

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.weights1, self.input.T))
        self.output = ReLU(np.dot(self.layer1.T, self.weights2))

    def backpropagation(self):
        delta2 = (self.y - self.output) * \
            ReLU_derivative(self.output)
        d_weights2 = self.eta * np.dot(self.layer1, delta2)

        delta1 = sigmoid_derivative(self.layer1) * \
            np.dot(self.weights2, delta2.T)
        d_weights1 = self.eta * np.dot(delta1, self.input)

        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def prediction(self, x):
        layer_pred = sigmoid(np.dot(self.weights1, x.T))
        return ReLU(np.dot(layer_pred.T, self.weights2))


fig, ax = plt.subplots()
ln, = plt.plot([], [], 'ro', markersize=3)
step = [1]


def init():
    ax.set_xlim(-55, 55)
    ax.set_ylim(-100, 3000)
    return ln,


def update(frame):
    for _ in range(100):
        nn.feedforward()
        nn.backpropagation()
    print("Krok uczenia:", round(step[0]*0.1, 1), "k")
    print("Blad (training set):")
    print(np.mean((y - nn.output)**2))

    pred = nn.prediction(x_test)
    print("Blad (testing set):")
    print(np.mean((y_t - pred)**2))
    step[0] += 1

    ln.set_data(x_t, pred)
    return ln,


if __name__ == "__main__":
    x = np.reshape(np.linspace(-50, 50, 26), (26, 1))
    y = x**2
    x_s = np.ones((26, 1))
    x = np.concatenate((x, x_s), axis=1)  # BIAS
    nn = NeuralNetwork(x, y)

    x_t = np.reshape(np.linspace(-50, 50, 101), (101, 1))
    y_t = x_t**2
    x_s = np.ones((101, 1))
    x_test = np.concatenate((x_t, x_s), axis=1)  # BIAS

    plt.plot(x_t, y_t, 'bo', markersize=5)
    ani = FuncAnimation(fig, update, frames=20, init_func=init)
    plt.show()
