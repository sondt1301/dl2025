from labwork4.neuron import Neuron

class Layer:
    def __init__(self, weight, bias_weight, input):
        self.neurons = [Neuron(weight[i], bias_weight[i], input) for i in range(len(weight))]
        self.layer_node_values = []

    def activate_layer(self):
        i = 1
        for neuron in self.neurons:
            print(f'Neuron {i}')
            i += i
            self.layer_node_values.append(neuron.activate())
        return self.layer_node_values
