import numpy as np
import functions
import random

class Network(object):
    def __init__(self, layers):
        self.depth = len(layers)  # depth of the layer
        self.layers = layers  # list of number of units for layers
        self.activation = functions.sigmoid  # activation function
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]  # random initialization of biases
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]  # from first to second zipped to obtain
         #weights going

    def feedforward(self, input):  # x = np.array([15,11]).reshape(2,1)
        output = input
        for W, b in zip(self.weights, self.biases):
            output = self.activation(np.dot(W, output) + b)

        return output

    def SGD(self, trainingData, epochs, batchSize, eta, testData = None):
        n = len(trainingData)

        for jj in range(epochs):
            random.shuffle(trainingData)
            miniBatches = [trainingData[ii: ii + batchSize] for ii in range(0, n, batchSize)]
            for miniBatch in miniBatches:
                self.update(miniBatch, eta)

            if testData:
                print("Epoch {0}: {1} / {2}").format(jj, self.evaluate(testData), len(testData))
            else:
                print("Epoch {0} complete".format(jj))


    def update(self, miniBatch, eta):
        deltaWeights = [np.zeros(W.shape) for W in self.weights]
        deltaBiases = [np.zeros(b.shape) for b in self.biases]

        for x, y in miniBatch:  # for every point obtaining the gradient
            biases, weights = self.EBP(x, y)
            deltaWeights = [old + new for old, new in zip(deltaWeights, weights)]
            deltaBiases = [old + new for old, new in zip(deltaBiases, biases)]

        size = len(miniBatch)
        self.biases = [b - (eta/size)*delta for b, delta in zip(self.biases, deltaBiases)]
        self.weights = [W - (eta/size)*delta for W, delta in zip(self.weights, deltaWeights)]


    def EBP(self, x, y):
        z = []  # list to store all the z vectors, layer by layer
        activation = x
        activations = [activation]  # we start from the input

        # feedforward
        for b, w in zip(self.biases, self.weights):
            temp = np.dot(w, activation) + b
            z.append(temp)
            activation = self.activation(temp)
            activations.append(activation)

        deltas = [np.zeros(b.shape) for b in self.biases]
        weights = [np.zeros(W.shape) for W in self.weights]
        biases = [np.zeros(b.shape) for b in self.biases]

        # last layer
        deltas[-1] = self.difference(activations[-1], y)*functions.derivativeSigmoid(z[-1])
        weights[-1] = np.dot(deltas[-1], activations[-2].transpose()) # check the table
        biases[-1] = deltas[-1]

        # backward
        for ii in range(self.depth - 3, -1, -1):
            deltas[ii] = np.dot(self.weights[ii+1].transpose(), deltas[ii + 1])*functions.derivativeSigmoid(z[ii])
            weights[ii] = np.dot(deltas[ii], activations[ii].transpose()) # activations are on the same level
            biases[ii] = deltas[ii]
        # for ii in range(2, self.depth): # -1 to get proper values
        #     deltas[-ii] = np.dot(self.weights[-ii+1].transpose(), deltas[-ii + 1])*functions.derivativeSigmoid(z[-ii])
        #     weights[-ii] = np.dot(deltas[-ii], activations[-ii-1].transpose())
        #     biases[-ii] = deltas[-ii]

        return biases, weights  # arrays of weights and biases

    def difference(self, output, y):
        diff = output - y
        return diff

    def evaluate(self, testData):
        results = [(np.argmax(self.feedforward(x)), y) for (x, y) in testData]
        score = sum(int(x == y) for (x, y) in results)  # check this one
        return score

if __name__ == "__main__":
    pass
