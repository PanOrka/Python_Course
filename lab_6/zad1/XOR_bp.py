import numpy


def sigmoid(x):
    return 1.0/(1+numpy.exp(-x))


def sigmoid_derivative(x):
    return x*(1.0-x)


def ReLU(x):
    b = x.copy()
    b[b < 0.0] = 0.0
    return b


def ReLU_derivative(x):
    b = x.copy()
    b[b < 0.0] = 0.0
    b[b >= 0.0] = 1.0
    return b


class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = numpy.random.rand(4, self.input.shape[1])
        self.weights2 = numpy.random.rand(1, 4)
        self.y = y
        self.output = numpy.zeros(self.y.shape)
        self.eta = 0.5

    def feedforward(self):
        self.layer1 = sigmoid(numpy.dot(self.input, self.weights1.T))
        self.output = ReLU(numpy.dot(self.layer1, self.weights2.T))

    def backprop(self):
        delta2 = (self.y - self.output) * \
            ReLU_derivative(self.output)
        d_weights2 = self.eta * numpy.dot(delta2.T, self.layer1)

        delta1 = sigmoid_derivative(self.layer1) * \
            numpy.dot(delta2, self.weights2)
        d_weights1 = self.eta * numpy.dot(delta1.T, self.input)

        self.weights1 += d_weights1
        self.weights2 += d_weights2


if __name__ == "__main__":
    X = numpy.array([[0, 0, 1],   # jedynka sluzy tutaj jako bias
                     [0, 1, 1],   # przedstawiona jako 1*w_b zamiast b
                     [1, 0, 1],   # dla neuronow w hidden layer zostana dodane
                     [1, 1, 1]])  # w_b pelniace role biasu dla kazdego z nich
    y = numpy.array([[0], [1], [1], [0]])
    nn = NeuralNetwork(X, y)

    for i in range(5000):
        nn.feedforward()
        nn.backprop()

    print(nn.output)
    print(nn.weights1)
    print(nn.weights2)
