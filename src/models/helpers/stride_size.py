

def get_encoder_stride_sizes(image_size: int, len_hidden_dims: int):
    """
    Returns a list of stride sizes for the encoder.
    """
    stride_sizes = []
    count_div_2 = count_div_by_2(image_size)
    for i in range(len_hidden_dims):
        stride_sizes.append(2 if count_div_2 > 0 else 1)
        count_div_2 -= 1

    return stride_sizes

def get_decoder_stride_sizes(image_size: int, len_hidden_dims: int):
    # return the reverse of the encoder stride sizes
    stride_sizes = get_encoder_stride_sizes(image_size, len_hidden_dims)
    stride_sizes.reverse()
    return stride_sizes
        


def count_div_by_2(num: int):
    """
    Returns the number of times num can be divided by 2.
    """
    count = 0
    while num % 2 == 0 and num > 0:
        count += 1
        num //= 2
    return count
