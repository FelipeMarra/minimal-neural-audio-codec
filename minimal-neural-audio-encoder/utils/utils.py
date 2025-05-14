def get_padding(kernel:int, stride:int=1) -> int:
    """
        In the non-streamable setup, we use for each convolution a total padding of K - S, split equally 
        before the first time step and after the last one (with one more before if K - S is odd).
    """
    padding = kernel - stride
    padding = padding + 1 if padding % 2 != 0 else padding # +1 if kernel-stride is odd
    padding = padding // 2 # split equally before the first time step and after the last one

    return padding

def get_conv_out(in_t:int, kernel:int, stride:int=1, padding:int=0) -> int:
    return ((in_t - kernel + 2*padding) // stride) + 1

def get_conv_transpose_out(in_t:int, kernel:int, stride:int=1, padding:int=0) -> int:
    return (in_t - 1) * stride - 2*padding + kernel