from numpy import exp, array, random, dot

class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs):
        self.synaptic_weights = 2 * random.random((number_of_inputs, number_of_neurons)) - 1

class NeuralNetwork():
    def __init__(self,data,depth,neurons):
        random.seed(1)
        self.setTrainingData(data)
        self.layer = []
        for d in range(depth-1):
            self.layer.append(NeuronLayer(neurons,len(self.training_inputs[0])))
        self.layer.append(NeuronLayer(1,neurons))

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def getTrainingData(self):
        return [[i,o] for i,o in zip(training_inputs,training_outputs)]

    def setTrainingData(self,data):
        self.training_inputs = array([e[0] for e in data])
        self.training_outputs = array([[e[1] for e in data]]).T

    def train(self, data, iterations):
        for n in range(iterations):
            thoughts = self.think(training_set_inputs,showHidden=True)

            layer_delta = {}
            layer_adjustment = {}
            layer_error = {}
            layer_error[len(thoughts)-1] = training_set_outputs - thoughts[-1]

            for t in reversed(range(1,len(thoughts))):
                layer_delta[t] = layer_error[t] * self.__sigmoid_derivative(thoughts[t])
                layer_adjustment[t] = thoughts[t-1].T.dot(layer_delta[t])

                layer_error[t-1] = layer_delta[t].dot(self.layer[t].synaptic_weights.T)

            layer_delta[0] = layer_error[0] * self.__sigmoid_derivative(thoughts[0])
            layer_adjustment[0] = training_set_inputs.T.dot(layer_delta[0])

            for k in layer_adjustment.keys():
                self.layer[k].synaptic_weights += layer_adjustment[k]

    def study(self, inputs):
        thoughts = [array(inputs)]
        for e in self.layer:
            thoughts.append(self.__sigmoid(dot(iterative, e.synaptic_weights)))
        return thoughts

    def think(self, inputs):
        iterative = array(inputs)
        for e in self.layer:
            iterative = self.__sigmoid(dot(iterative, e.synaptic_weights))
        return iterative


if __name__ == "__main__":
    data = [[[0, 0, 1],0]
            [[0, 1, 1],1]
            [[1, 0, 1],1]
            [[0, 1, 0],1]
            [[1, 0, 0],1]
            [[1, 1, 1],0]
            [[0, 0, 0],0]])

    nn = NeuralNetwork(data,2,4)
    nn.train(100)

    print(nn.think([1, 1, 0]))
