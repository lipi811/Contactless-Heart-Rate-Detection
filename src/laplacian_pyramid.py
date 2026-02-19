import cv2

def build_laplacian_pyramid(gaussian_pyramid):
    """
    Build Laplacian pyramid from Gaussian pyramid.

    Parameters:
        gaussian_pyramid (list): List of Gaussian pyramid images

    Returns:
        laplacian_pyramid (list): Laplacian pyramid images
    """
    laplacian_pyramid = []

    for i in range(len(gaussian_pyramid) - 1):
        size = (
            gaussian_pyramid[i].shape[1],
            gaussian_pyramid[i].shape[0]
        )

        expanded = cv2.pyrUp(
            gaussian_pyramid[i + 1],
            dstsize=size
        )

        laplacian = cv2.subtract(gaussian_pyramid[i], expanded)
        laplacian_pyramid.append(laplacian)

    return laplacian_pyramid
