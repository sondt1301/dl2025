import random

from CNN.Helper import create_zeros_list

class Dense:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        # Weights: [output_neuron_idx][input_neuron_idx]
        self.weights = [[random.uniform(-1.0, 1.0) for _ in range(input_size)] for _ in range(output_size)]
        self.biases = [random.uniform(-1.0, 1.0) for _ in range(output_size)]
        self.last_input = None

    def forward(self, input_1d_list):
        self.last_input = input_1d_list
        output_1d_list = [0.0] * self.output_size
        for out_idx in range(self.output_size):
            current_sum = 0.0
            for in_idx in range(self.input_size):
                current_sum += self.weights[out_idx][in_idx] * input_1d_list[in_idx]
            output_1d_list[out_idx] = current_sum + self.biases[out_idx]
        return output_1d_list

    # d_out_1d_list = dL/dz, size = output_size
    def backward(self, d_out_1d_list, lr):
        d_weights = create_zeros_list((self.output_size, self.input_size))
        d_biases = [0.0] * self.output_size
        dL_dInput_to_this_dense = [0.0] * self.input_size

        for out_idx in range(self.output_size):
            d_output_element = d_out_1d_list[out_idx]
            d_biases[out_idx] = d_output_element

            for in_idx in range(self.input_size):
                # dL/dWeight_ji = delta_j * Activation_i_from_prev_layer
                d_weights[out_idx][in_idx] = d_output_element * self.last_input[in_idx]

                # Add on dL/dActivation_i_from_prev_layer
                # In neural network labwork, this is delta
                dL_dInput_to_this_dense[in_idx] += d_output_element * self.weights[out_idx][in_idx]

        # Update weights and biases
        for r in range(self.output_size):
            for c in range(self.input_size):
                self.weights[r][c] -= lr * d_weights[r][c]
            self.biases[r] -= lr * d_biases[r]

        return dL_dInput_to_this_dense