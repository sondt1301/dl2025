from labwork5.neuron import Neuron

class Layer:
    def __init__(self, layer_weight, layer_bias_weight, layer_input):
        self.neurons = [Neuron(layer_weight[i], layer_bias_weight[i], layer_input) for i in range(len(layer_weight))]
        self.layer_node_values = []

    def activate_layer(self):
        self.layer_node_values = []
        neuron_counter = 1
        for neuron in self.neurons:
            neuron_counter += neuron_counter
            self.layer_node_values.append(neuron.activate())
        return self.layer_node_values
