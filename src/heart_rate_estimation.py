import numpy as np

def estimate_heart_rate(freqs, magnitude, low=0.75, high=4.0):
    """
    Estimate heart rate in BPM from FFT magnitude spectrum.

    Parameters:
        freqs (numpy.ndarray): Frequency bins (Hz)
        magnitude (numpy.ndarray): FFT magnitude
        low (float): Lower frequency bound (Hz)
        high (float): Upper frequency bound (Hz)

    Returns:
        bpm (float): Estimated heart rate in BPM
    """

    # Select valid heart-rate frequency range
    valid_idx = np.where((freqs >= low) & (freqs <= high))

    if len(valid_idx[0]) == 0:
        return None

    peak_freq = freqs[valid_idx][np.argmax(magnitude[valid_idx])]
    bpm = peak_freq * 60.0

    return bpm
