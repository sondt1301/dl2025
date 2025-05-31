import random

from CNN.Helper import create_zeros_list

class Conv2D:
    def __init__(self, filter_size, input_depth=1, stride=1, padding=0):
        # Size of the square filter
        self.filter_size = filter_size
        self.input_depth = input_depth
        self.stride = stride
        self.padding = padding
        self.filter = [[[random.uniform(-1.0, 1.0) for _ in range(filter_size)] for _ in range(filter_size)] for _ in range(input_depth)]
        self.bias = random.uniform(-1.0, 1.0)
        self.last_input = None
        self.output_shape = None

    # Add a border of 0 around a 2D input based on padding
    def pad_input(self, input_channel):
        if self.padding == 0:
            return input_channel
        h = len(input_channel)
        w = len(input_channel[0])
        padded_h = h + 2 * self.padding
        padded_w = w + 2 * self.padding
        padded = create_zeros_list((padded_h, padded_w))
        for r in range(h):
            for c in range(w):
                padded[r + self.padding][c + self.padding] = input_channel[r][c]
        return padded

    def forward(self, input_volume):
        self.last_input = input_volume

        input_h_orig = len(input_volume[0])
        input_w_orig = len(input_volume[0][0])

        input_h = input_h_orig + 2 * self.padding
        input_w = input_w_orig + 2 * self.padding

        # Compute output dimension out = ((in - filter) / stride) + 1
        out_h = (input_h - self.filter_size) // self.stride + 1
        out_w = (input_w - self.filter_size) // self.stride + 1

        # Create an empty output map as a placeholder
        self.output_shape = (out_h, out_w)
        output_map = create_zeros_list((out_h, out_w))

        # Pad each input channel individually if padding > 0
        padded_input_volume = []
        if self.padding > 0:
            for d in range(self.input_depth):
                padded_input_volume.append(self.pad_input(input_volume[d]))
        else:
            padded_input_volume = input_volume

        # Loop over each output pixel position.
        for r_out in range(out_h):
            for c_out in range(out_w):
                current_sum = 0.0
                for d_in in range(self.input_depth):
                    for fr in range(self.filter_size):
                        for fc in range(self.filter_size):
                            # Calculate corresponding input coordinates
                            input_r = r_out * self.stride + fr
                            input_c = c_out * self.stride + fc
                            current_sum += self.filter[d_in][fr][fc] * padded_input_volume[d_in][input_r][input_c]
                # Assign result to each pixel in output map
                output_map[r_out][c_out] = current_sum + self.bias
        return output_map

    # Calculate 3 gradients:
    # filter weights,
    # bias,
    # input of the layer (so that the previous layer can take it as the gradient of its own output)
    def backward(self, d_out, lr):
        out_h, out_w = self.output_shape

        d_filter = create_zeros_list((self.input_depth, self.filter_size, self.filter_size))
        d_bias_val = 0.0

        # Pad input for gradient
        padded_last_input_volume = []
        if self.padding > 0:
            for d in range(self.input_depth):
                padded_last_input_volume.append(self.pad_input(self.last_input[d]))
        else:
            padded_last_input_volume = self.last_input

        # Calculate d_filter and d_bias
        # Loop over each pixel in the output
        for r_out in range(out_h):
            for c_out in range(out_w):
                d_output_element = d_out[r_out][c_out]
                # dL/db = dL/dy
                d_bias_val += d_output_element

                # Loop over each pixel in the filter
                for d_in in range(self.input_depth):
                    for fr in range(self.filter_size):
                        for fc in range(self.filter_size):
                            input_r = r_out * self.stride + fr
                            input_c = c_out * self.stride + fc
                            # dL/dw = input_x * dL/dy
                            d_filter[d_in][fr][fc] += padded_last_input_volume[d_in][input_r][input_c] * d_output_element


        # Consider only case when stride=1, padding=0 in forward
        # Initialize dL_dInput
        orig_input_h = len(self.last_input[0])
        orig_input_w = len(self.last_input[0][0])
        dL_dInput_volume = create_zeros_list((self.input_depth, orig_input_h, orig_input_w))

        # Loop over each pixel in the output
        for r_out in range(out_h):
            for c_out in range(out_w):
                d_output_element = d_out[r_out][c_out]

                # Loop over each pixel in the filter
                for d_in in range(self.input_depth):
                    for fr in range(self.filter_size):
                        for fc in range(self.filter_size):
                            input_r_padded = r_out * self.stride + fr
                            input_c_padded = c_out * self.stride + fc

                            # Convert back to original input index if there was padding
                            input_r_orig = input_r_padded - self.padding
                            input_c_orig = input_c_padded - self.padding

                            # Ensure positions of the input volume < Initial volume size
                            if 0 <= input_r_orig < orig_input_h and 0 <= input_c_orig < orig_input_w:
                                # dL/dInput = input_x * dL/dy
                                dL_dInput_volume[d_in][input_r_orig][input_c_orig] += self.filter[d_in][fr][fc] * d_output_element

        # Update filter and bias
        for d in range(self.input_depth):
            for fr in range(self.filter_size):
                for fc in range(self.filter_size):
                    self.filter[d][fr][fc] -= lr * d_filter[d][fr][fc]
        self.bias -= lr * d_bias_val

        return dL_dInput_volume
