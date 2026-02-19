import numpy as np

def amplify_signal(signal_data, gain=10):
    """
    Amplify rPPG signal.

    Parameters:
        signal_data (numpy.ndarray): Filtered rPPG signal
        gain (float): Amplification factor

    Returns:
        amplified_signal (numpy.ndarray)
    """
    return signal_data * gain
