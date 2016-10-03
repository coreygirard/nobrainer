from numpy import exp, array, random, dot

class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs):
        self.synaptic_weights = 2 * random.random((number_of_inputs, number_of_neurons)) - 1

class ShallowNetwork():
    def __init__(self,data):
        random.seed(1)
        self.setTrainingData(data)
        self.layer = NeuronLayer(1,len(self.training_inputs[0]))

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
            self.layer.synaptic_weights += adjustment

    def think(self, inputs):
        return self.__sigmoid(dot(array(inputs), self.layer.synaptic_weights))

class DeepNetwork():
    def __init__(self,data,depth,neurons):
        random.seed(1)
        self.setTrainingData(data)
        self.layer = []

        assert(depth > 1)

        self.layer.append(NeuronLayer(neurons,len(self.training_inputs[0])))
        for t in range(depth-2):
            self.layer.append(NeuronLayer(neurons,neurons))
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

    def train(self, iterations):
        for n in range(iterations):
            thoughts = self.study(self.training_inputs)

            e = len(self.layer)-1
            layer_error = {e:self.training_outputs - thoughts[e]}
            layer_delta = {e:layer_error[e] * self.__sigmoid_derivative(thoughts[e])}
            for t in reversed(range(e)):
                layer_error[t] = layer_delta[t+1].dot(self.layer[t+1].synaptic_weights.T)
                layer_delta[t] = layer_error[t] * self.__sigmoid_derivative(thoughts[t])

            layer_adjustment = {0:self.training_inputs.T.dot(layer_delta[0])}
            for t in range(1,len(thoughts)):
                layer_adjustment[t] = thoughts[t-1].T.dot(layer_delta[t])

            for t in range(len(self.layer)):
                self.layer[t].synaptic_weights += layer_adjustment[t]

    def study(self, inputs):
        thoughts = [array(inputs)]
        for e in self.layer:
            thoughts.append(self.__sigmoid(dot(thoughts[-1], e.synaptic_weights)))
        return thoughts[1:]

    def think(self, inputs):
        iterative = array(inputs)
        for e in self.layer:
            iterative = self.__sigmoid(dot(iterative, e.synaptic_weights))
        return iterative


if __name__ == "__main__":
    data = [[[0, 0, 1],0],
            [[0, 1, 1],1],
            [[1, 0, 1],1],
            [[0, 1, 0],1],
            [[1, 0, 0],1],
            [[1, 1, 1],0],
            [[0, 0, 0],0]]

    nn = DeepNetwork(data,2,4)
    nn.train(60000)

    print(nn.think([1, 1, 0]))
