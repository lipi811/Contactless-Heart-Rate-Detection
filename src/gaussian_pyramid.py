import cv2

def build_gaussian_pyramid(image, levels=3):
    """
    Build Gaussian pyramid from an image.

    Parameters:
        image (numpy.ndarray): Input image (face ROI)
        levels (int): Number of pyramid levels

    Returns:
        pyramid (list): Gaussian pyramid images
    """
    pyramid = [image]

    for _ in range(levels):
        image = cv2.pyrDown(image)
        pyramid.append(image)

    return pyramid
