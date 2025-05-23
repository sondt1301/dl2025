import math

class Neuron:
    def __init__(self, weight, bias_weight, input):
        self.weight = weight
        self.bias_weight = bias_weight
        self.input = input
        self.linear_sum = 0.0
        self.activation = 0.0
        self.delta = 0.0

    def get_linear_sum(self):
        sum = 0
        for i in range(len(self.weight)):
            sum += self.input[i] * self.weight[i]
        self.linear_sum = sum + self.bias_weight
        return self.linear_sum

    def sigmoid(self, linear_sum):
        if linear_sum < -700: return 0.0
        if linear_sum > 700:  return 1.0

        sigmoid = 1 / (1 + math.exp(-linear_sum))
        return sigmoid

    def activate(self):
        self.get_linear_sum()
        self.activation = self.sigmoid(self.linear_sum)
        return self.activation
