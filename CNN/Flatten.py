from CNN.Helper import create_zeros_list

class Flatten:
    # Store the input shape
    def __init__(self):
        self.input_depth = None
        self.input_height = None
        self.input_width = None

    def forward(self, input_3d_volume):
        self.input_depth = len(input_3d_volume)
        self.input_height = len(input_3d_volume[0])
        self.input_width = len(input_3d_volume[0][0])

        # Flatten to 1D
        flat_list = []
        for d_channel in input_3d_volume:
            for row in d_channel:
                flat_list.extend(row)
        return flat_list

    # d_out_flat_list: 1D gradient list from the next layer
    def backward(self, d_out_flat_list):
        # Reconstruct to 3D
        dL_dinput_3d_volume = create_zeros_list((self.input_depth, self.input_height, self.input_width))
        idx = 0
        for d in range(self.input_depth):
            for r in range(self.input_height):
                for c in range(self.input_width):
                    dL_dinput_3d_volume[d][r][c] = d_out_flat_list[idx]
                    idx += 1
        return dL_dinput_3d_volume
