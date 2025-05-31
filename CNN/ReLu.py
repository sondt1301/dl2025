from CNN.Helper import create_zeros_list

# Negative values are set to 0, positives pass through
class ReLU:
    def __init__(self):
        # Stores the input from the forward pass
        self.last_input = None

    def forward(self, input_2d):
        self.last_input = input_2d
        output_2d = create_zeros_list((len(input_2d), len(input_2d[0])))
        for r in range(len(input_2d)):
            for c in range(len(input_2d[0])):
                output_2d[r][c] = max(0.0, input_2d[r][c])
        return output_2d

    # d_out_2d: gradient of the loss with respect to ReLU input
    def backward(self, d_out_2d):
        dL_dinput_2d = create_zeros_list((len(d_out_2d), len(d_out_2d[0])))
        for r in range(len(d_out_2d)):
            for c in range(len(d_out_2d[0])):
                # Ensures only neurons that were positive in forward pass propagate gradient backward
                if self.last_input[r][c] > 0:
                    dL_dinput_2d[r][c] = d_out_2d[r][c]
                else:
                    dL_dinput_2d[r][c] = 0.0
        return dL_dinput_2d
