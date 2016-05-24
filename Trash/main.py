import loadMNIST
import network
import test22

if __name__ == "__main__":
    trainingData, validationData, testData = loadMNIST.loadData()
    net = network.Network([784, 30, 10])
    net.SGD(trainingData, 30, 10, 3.0, testData)  # trainingData, epochs, batchSize, eta, testData = None)

    #net.evaluate(testData)
    #net.activation(1)



