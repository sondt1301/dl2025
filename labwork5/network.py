import math

from labwork5.layer import Layer

class Network:
    def __init__(self, all_layer_weight, all_layer_bias_weight, initial_input):
        self.layers = [Layer(all_layer_weight[i], all_layer_bias_weight[i], initial_input) for i in range(len(all_layer_weight))]
        self.output = []

    def feedforward(self, layer_input):
        layer_input = layer_input
        for i, layer in enumerate(self.layers):
            for neuron in layer.neurons:
                neuron.input = layer_input
            layer_output_values = layer.activate_layer()
            layer_input = layer_output_values
        self.output = layer_input
        return self.output

    # LOSS FUNCTION
    def loss(self, linear_sum, y):
        loss = -y * linear_sum + math.log(1 + math.exp(linear_sum))
        return loss

    def train(self, training_x, training_y, rate, epochs):
        num_samples = len(training_x)
        for epoch in range(epochs):
            total_loss = 0
            for i in range (num_samples):
                x_sample = training_x[i]
                y_sample = training_y[i]

                # Feedforward
                print(f"\n--- Epoch {epoch + 1}/{epochs}, Sample {i + 1}/{num_samples} ---")
                network_output = self.feedforward(x_sample)

                # Loss Calculation
                sample_loss = 0
                output_layer = self.layers[-1]
                for i, output_neuron in enumerate(output_layer.neurons):
                    y = (y_sample[i])
                    linear_sum = output_neuron.linear_sum

                    neuron_loss = self.loss(linear_sum, y)
                    sample_loss += neuron_loss

                    # Delta of the output layer
                    output_neuron.delta = output_neuron.activation - y
                total_loss += sample_loss
                print(f"Input: {x_sample}, First Run: {[float(f'{o:.4f}') for o in network_output]}, Target: {y_sample}, Logistic Loss: {sample_loss:.4f}")

                # Delta of hidden layer
                for i in range(len(self.layers) -2, -1, -1):
                    current_layer = self.layers[i]
                    next_layer = self.layers[i + 1]
                    for current_idx, current_neuron in enumerate(current_layer.neurons):
                        sum_err_delta = 0
                        for next_idx, next_neuron in enumerate(next_layer.neurons):
                            sum_err_delta += next_neuron.delta * next_neuron.weight[current_idx]
                        sigmoid_derivative = current_neuron.activation * (1 - current_neuron.activation)
                        current_neuron.delta = sum_err_delta * sigmoid_derivative

                # Weight Update
                for layer_idx, layer in enumerate(self.layers):
                    for neuron in layer.neurons:
                        for weight_idx in range(len(neuron.weight)):
                            weight_gradient = neuron.delta * neuron.input[weight_idx]
                            neuron.weight[weight_idx] -= rate * weight_gradient

                        bias_gradient = neuron.delta
                        neuron.bias_weight -= rate * bias_gradient

            avg_epoch_loss = total_loss / num_samples
            print(f"--- Epoch {epoch + 1}/{epochs} Summary ---")
            print(f"Average Loss (Logistic Form): {avg_epoch_loss:.4f}")
            if avg_epoch_loss < 0.01 and epoch > 10:
                print("Loss is very low.")
                break
        print("--- Training Complete ---")

    def predict_threshold(self, input_sample):
        raw_outputs = self.feedforward(input_sample)
        threshold_outputs = [1 if out > 0.5 else 0 for out in raw_outputs]
        return threshold_outputs