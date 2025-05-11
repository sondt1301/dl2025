from labwork4.layer import Layer

class Network:
    def __init__(self, weight, bias_weight, input):
        self.layers = [Layer(weight[i], bias_weight[i], input) for i in range(len(weight))]
        self.output = []

    def get_result(self):
        i = 1
        for layer in self.layers:
            print(f'Layer {i}')
            i += i
            layer.layer_node_values = layer.activate_layer()
            new_input_for_next_layer = layer.layer_node_values
            for next_layer in self.layers[1:]:
                for neuron in next_layer.neurons:
                    neuron.input = new_input_for_next_layer
            self.output = new_input_for_next_layer

        return self.output
