import numpy
import scipy.special

class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        # sigmoid 
        self.activation_function = lambda x : scipy.special.expit(x)

        # init weights
        self.weights_input_hidden = numpy.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.weights_hidden_output = numpy.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.output_nodes, self.hidden_nodes))
    
    def train(self, input_lists, target_lists):
        inputs = numpy.array(input_lists, ndmin=2).T
        targets = numpy.array(target_lists, ndmin=2).T

        # Feedforward
        hidden_layer_inputs = numpy.dot(self.weights_input_hidden, inputs)
        hidden_layer_ouputs = self.activation_function(hidden_layer_inputs)
        final_layer_inputs = numpy.dot(self.weights_hidden_output, hidden_layer_ouputs)
        final_layer_outputs = self.activation_function(final_layer_inputs)

        # Backpropagation
        output_layer_errors = targets - final_layer_outputs
        hidden_layer_errors = numpy.dot(self.weights_hidden_output.T, output_layer_errors)
        # Update weights
        self.weights_hidden_output += self.learning_rate * numpy.dot((output_layer_errors * final_layer_outputs * (1.0 - final_layer_outputs)), 
                                                                     numpy.transpose(hidden_layer_ouputs))
        self.weights_input_hidden += self.learning_rate * numpy.dot((hidden_layer_errors * hidden_layer_ouputs * (1.0 - hidden_layer_ouputs)),
                                                                     numpy.transpose(inputs))

        

    def query(self, input_lists):
        inputs = numpy.array(input_lists, ndmin=2).T

        # Feedforward
        hidden_layer_inputs = numpy.dot(self.weights_input_hidden, inputs)
        hidden_layer_ouputs = self.activation_function(hidden_layer_inputs)
        final_layer_inputs = numpy.dot(self.weights_hidden_output, hidden_layer_ouputs)
        final_layer_outputs = self.activation_function(final_layer_inputs)

        return final_layer_outputs