import numpy
from neural_network import NeuralNetwork

NR_INPUT_NODES = 784
NR_HIDDEN_NODES = 100
NR_OUTPUT_NODES = 10

LEARNING_RATE = 0.3

neural_network = NeuralNetwork(NR_INPUT_NODES, NR_HIDDEN_NODES, NR_OUTPUT_NODES, LEARNING_RATE)

training_data_file = open('dataset/training/mnist_train_100.csv')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the network
for training_data in training_data_list:
    all_values = training_data.split(',')
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(NR_OUTPUT_NODES) + 0.01
    targets[int(all_values[0])] + 0.99
    neural_network.train(inputs, targets)

