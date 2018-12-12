import numpy
from neural_network import NeuralNetwork

NR_INPUT_NODES = 784
NR_HIDDEN_NODES = 200
NR_OUTPUT_NODES = 10

LEARNING_RATE = 0.1

EPOCH = 5

neural_network = NeuralNetwork(NR_INPUT_NODES, NR_HIDDEN_NODES, NR_OUTPUT_NODES, LEARNING_RATE)

training_data_file = open('dataset/training/mnist_train.csv')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the network
for training_data in training_data_list:
    all_values = training_data.split(',')
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(NR_OUTPUT_NODES) + 0.01
    targets[int(all_values[0])] + 0.99
    neural_network.train(inputs, targets)

test_data_file = open('dataset/test/mnist_test.csv')
test_data_list = test_data_file.readlines()
test_data_file.close()

# query the network
scorecard = []
for e in range(EPOCH):
    for test_data in test_data_list:
        all_values = test_data.split(',')
        correct_label = int(all_values[0])
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = neural_network.query(inputs)
        network_label = numpy.argmax(outputs)
        print('correct_label: ', correct_label, ' network_answer: ', network_label)
        if (network_label == correct_label):
            scorecard.append(1)
        else:
            scorecard.append(0)

scorecard_array = numpy.asarray(scorecard)
print('network performance = ', scorecard_array.sum() / scorecard_array.size)


