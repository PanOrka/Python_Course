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
    def __init__(self, x, y, func, d_func,
                 hidden_layer_size_1=5, hidden_layer_size_2=3):
        self.input = x
        self.weights1 = np.random.rand(hidden_layer_size_1, 2)
        self.weights2 = np.random.rand(hidden_layer_size_2, hidden_layer_size_1 + 1)
        self.weights3 = np.random.rand(hidden_layer_size_2, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)
        self.eta = 0.000000001  # 0.00000001 max eta
        self.func = func
        self.d_func = d_func

    def feedforward(self):
        self.layer1 = self.func[0](np.dot(self.weights1, self.input.T))
        self.layer1_bias = np.concatenate(
            (self.layer1, np.ones((1, self.layer1.shape[1]))),
            axis=0
        )
        self.layer2 = self.func[1](np.dot(self.weights2, self.layer1_bias))
        self.output = self.func[2](np.dot(self.layer2.T, self.weights3))

    def backpropagation(self):
        delta3 = (self.y - self.output) * \
            self.d_func[2](self.output)
        d_weights3 = self.eta * np.dot(self.layer2, delta3)

        delta2 = self.d_func[1](self.layer2) * \
            np.dot(self.weights3, delta3.T)
        d_weights2 = self.eta * np.dot(delta2, self.layer1_bias.T)

        # Usuwamy ostatnia kolumne z weights2, poniewaz
        # ostatnia kolumna odpowiadala za wagi biasu
        # a bias nie bedzie uwzgledniany w pierwszych wagach
        # bo nie ma zadnego inputu
        delta1 = self.d_func[0](self.layer1) * \
            np.dot(self.weights2[:, :-1].T, delta2)
        d_weights1 = self.eta * np.dot(delta1, self.input)

        self.weights1 += d_weights1
        self.weights2 += d_weights2
        self.weights3 += d_weights3

    def prediction(self, x):
        layer1 = self.func[0](np.dot(self.weights1, x.T))
        layer1_bias = np.concatenate(
            (layer1, np.ones((1, layer1.shape[1]))),
            axis=0
        )
        layer2 = self.func[1](np.dot(self.weights2, layer1_bias))
        return self.func[2](np.dot(layer2.T, self.weights3))


if __name__ == "__main__":
    x = np.reshape(np.linspace(-50, 50, 26), (26, 1))
    y = x**2
    x_s = np.ones((26, 1))
    x = np.concatenate((x, x_s), axis=1)  # BIAS
    func = (sigmoid, ReLU, ReLU)
    d_func = (sigmoid_derivative,
              ReLU_derivative,
              ReLU_derivative)
    nn = NeuralNetwork(x, y, func, d_func)

    x_t = np.reshape(np.linspace(-50, 50, 101), (101, 1))
    y_t = x_t**2
    x_s = np.ones((101, 1))
    x_test = np.concatenate((x_t, x_s), axis=1)  # BIAS
    while True:
        for i in range(100000):
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
        print(nn.weights3)
