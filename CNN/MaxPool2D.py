from CNN.Helper import create_zeros_list

class MaxPool2D:
    def __init__(self, size=2, stride=None):
        self.size = size
        self.stride = stride if stride is not None else size
        self.last_input_shape = None
        self.max_indices = None

    def forward(self, input_2d):
        self.last_input_shape = (len(input_2d), len(input_2d[0]))
        h, w = self.last_input_shape

        # Define h and w of the output
        out_h = (h - self.size) // self.stride + 1
        out_w = (w - self.size) // self.stride + 1

        output_2d = create_zeros_list((out_h, out_w))
        self.max_indices = create_zeros_list((out_h, out_w))

        # Scans each pooling window
        for r_out in range(out_h):
            for c_out in range(out_w):
                max_val = float('-inf')
                max_pos_in_input = (-1, -1)
                for pr in range(self.size):
                    for pc in range(self.size):
                        input_r = r_out * self.stride + pr
                        input_c = c_out * self.stride + pc

                        if input_r < h and input_c < w:
                            val = input_2d[input_r][input_c]
                            # Check max value and store position
                            if val > max_val:
                                max_val = val
                                max_pos_in_input = (input_r, input_c)
                output_2d[r_out][c_out] = max_val
                self.max_indices[r_out][c_out] = max_pos_in_input
        return output_2d

    # Backpropagate gradients only to the max values in each pooling region
    def backward(self, d_out_2d):
        h_in, w_in = self.last_input_shape
        dL_dinput_2d = create_zeros_list((h_in, w_in))

        out_h, out_w = len(d_out_2d), len(d_out_2d[0])

        for r_out in range(out_h):
            for c_out in range(out_w):
                if self.max_indices[r_out][c_out] != (-1, -1):
                    # Find the position of the max
                    max_r, max_c = self.max_indices[r_out][c_out]
                    # Sends the gradient back to that location
                    dL_dinput_2d[max_r][max_c] += d_out_2d[r_out][c_out]
        return dL_dinput_2d
