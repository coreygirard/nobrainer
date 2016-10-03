from numpy import exp, array, random, dot


class NeuralNetwork():
    def __init__(self,data):
        random.seed(1)
        self.setTrainingData(data)
        self.synaptic_weights = 2 * random.random((len(self.training_inputs[0]), 1)) - 1

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def getTrainingData(self):
        return [[i,o] for i,o in zip(self.training_inputs,self.training_outputs)]

    def setTrainingData(self,data):
        self.training_inputs = array([e[0] for e in data])
        self.training_outputs = array([[e[1] for e in data]]).T

    def train(self, iterations):
        for t in range(iterations):
            output = self.think(self.training_inputs)
            error = self.training_outputs - output
            adjustment = dot(self.training_inputs.T, error * self.__sigmoid_derivative(output))
            self.synaptic_weights += adjustment

    def think(self, inputs):
        return self.__sigmoid(dot(array(inputs), self.synaptic_weights))


if __name__ == "__main__":
    data = [[[0, 0, 1],0],
            [[1, 1, 1],1],
            [[1, 0, 1],1],
            [[0, 1, 1],0]]

    nn = NeuralNetwork(data)
    nn.train(10000)

    print(nn.think([1, 0, 0]))
