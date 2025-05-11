import ast
import random
from labwork4.network import Network

def read_data(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    data = []

    for line in lines:
        data.append(line.strip() if line.strip().isdigit() else ast.literal_eval(line.strip()))

    num_layers = int(data[0])
    print(data[1])
    num_neurons = [data[i] for i in range(1, num_layers + 1)]
    weight = data[num_layers + 1]
    bias_weight = data[num_layers + 2]

    return num_layers, num_neurons, weight, bias_weight

def check_fix_weight(file_path, input):
    num_layers, num_neurons, weight, bias_weight = read_data(file_path)

    network = Network(weight, bias_weight, input)

    output = network.get_result()
    print("================ RESULT WITH FIXED WEIGHT =================")
    print("=================================")
    print(f"Network output: {output}")

# def check_random_weight(file_path, input):
#     num_layers, num_neurons, _, _ = read_data(file_path)
#     weight = []
#     bias_weight = []
#     for i in range(num_layers-1):
#         neuron_weight = []
#         layer_weight = []
#         for i in range(int(num_neurons[i+1])):
#             neuron_weight.append(random.uniform(0, 1))
#             print('aaa', i)
#             layer_weight.append(neuron_weight)
#         weight.append(layer_weight)
#         bias_weight.append(random.uniform(0, 1))
#     print('weight', weight)
#     print('bias_weight', weight)
#     network = Network(weight, bias_weight, input)
#
#     output = network.get_result()
#     print("================ RESULT WITH RANDOM WEIGHT =================")
#     print("=================================")
#     print(f"Network output: {output}")


if __name__ == '__main__':
    file_path = "./input.txt"
    input = [1, 1]
    check_fix_weight(file_path, input)
    # check_random_weight(file_path, input)



