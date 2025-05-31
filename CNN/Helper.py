# Nested lists of zeros
def create_zeros_list(shape):
    if not shape:
        return 0.0
    return [create_zeros_list(shape[1:]) for _ in range(shape[0])]
