import random

from labwork5.network import Network

def read_data(file_path):
    x_data = []
    y_data = []

    with open(file_path, "r") as file:
        lines = file.readlines()

    # Remove header
    lines = lines[1:]

    for line in lines:
        x1, x2, y = line.strip().split(",")
        x_data.append([float(x1), float(x2)])
        y_data.append([float(y)])

    return x_data, y_data

def check_network(x_data, y_data):
    num_inputs = 2
    num_hidden_neurons = 2
    num_output_neurons = 1

    hidden_layer_weights = [[random.uniform(-1, 1) for _ in range(num_inputs)] for _ in range(num_hidden_neurons)]
    hidden_layer_biases = [random.uniform(-0.5, 0.5) for _ in range(num_hidden_neurons)]
    output_layer_weights = [[random.uniform(-1, 1) for _ in range(num_hidden_neurons)] for _ in range(num_output_neurons)]
    output_layer_biases = [random.uniform(-0.5, 0.5) for _ in range(num_output_neurons)]

    initial_weights = [hidden_layer_weights, output_layer_weights]
    initial_biases = [hidden_layer_biases, output_layer_biases]

    network = Network(initial_weights, initial_biases, x_data)

    print("Initial Network Weights:")
    for l_idx, layer_w in enumerate(network.layers):
        print(f" Layer {l_idx + 1} Weights:")
        for n_idx, neuron_w in enumerate(layer_w.neurons):
            print(f"  Neuron {n_idx + 1}: Weights={neuron_w.weight}, Bias={neuron_w.bias_weight:.4f}")

    learning_rate = 0.1
    epochs = 500

    network.train(x_data, y_data, learning_rate, epochs)

    print("\n================ TESTING TRAINED NETWORK =================")
    correct_predictions = 0
    for i, input_pattern in enumerate(x_data):
        print(f"\nTest Input: {input_pattern}")
        predicted_output_list = network.predict_threshold(input_pattern)
        predicted_val = predicted_output_list[0]
        expected_val = y_data[i][0]
        print(f"Input: {input_pattern}, Predicted (Thresholded): {predicted_val}, Expected: {expected_val}")
        if predicted_val == expected_val:
            correct_predictions += 1
            print("Result: CORRECT")
        else:
            print("Result: INCORRECT")
        print("-----------------------------------------------------")

    accuracy = correct_predictions / len(x_data)
    print(f"\nFinal Accuracy on data: {accuracy * 100:.2f}% ({correct_predictions}/{len(x_data)})")
    print("========================================================")

    print("\nFinal Network Weights After Training:")
    for l_idx, layer_w in enumerate(network.layers):
        print(f" Layer {l_idx + 1} Weights:")
        for n_idx, neuron_w in enumerate(layer_w.neurons):
            print(
                f"  Neuron {n_idx + 1}: Weights={[float(f'{w:.4f}') for w in neuron_w.weight]}, Bias={neuron_w.bias_weight:.4f}")

if __name__ == '__main__':
    file_path = "./input.csv"
    x_data, y_data = read_data(file_path)
    check_network(x_data, y_data)

    print("\n===================================================")
    file_path = "../loan2.csv"
    x_data, y_data = read_data(file_path)
    check_network(x_data, y_data)
