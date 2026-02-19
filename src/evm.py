import numpy as np

def amplify_laplacian(laplacian_pyramid, alpha=50):
    """
    Amplify Laplacian pyramid layers.

    Parameters:
        laplacian_pyramid (list): Laplacian pyramid images
        alpha (int): Amplification factor

    Returns:
        amplified_pyramid (list): Amplified Laplacian pyramid
    """
    amplified_pyramid = []

    for layer in laplacian_pyramid:
        amplified_layer = layer * alpha
        amplified_pyramid.append(amplified_layer)

    return amplified_pyramid
