import math

class Neuron:
    def __init__(self, weight, bias_weight, input):
        self.weight = weight
        self.bias_weight = bias_weight
        self.input = input

    def linear_sum(self):
        sum = 0
        print("Neuron input : ", self.input)
        print("Neuron weight: ", self.weight)
        print("Neuron bias weight: ", self.bias_weight)
        for i in range(len(self.weight)):
            sum += self.input[i] * self.weight[i]
        linear_sum = sum + self.bias_weight
        return linear_sum

    def sigmoid(self, linear_sum):
        sigmoid = 1 / (1 + math.exp(-linear_sum))
        return sigmoid

    def activate(self):
        linear_sum = self.linear_sum()
        sigmoid = 1 if self.sigmoid(linear_sum) > 0.5 else 0
        return sigmoid
