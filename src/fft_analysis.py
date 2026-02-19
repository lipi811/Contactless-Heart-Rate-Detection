import numpy as np

def compute_fft(signal_data, fps):
    """
    Compute FFT of amplified rPPG signal.

    Parameters:
        signal_data (numpy.ndarray): Amplified rPPG signal
        fps (float): Frames per second

    Returns:
        freqs (numpy.ndarray): Frequency bins (Hz)
        magnitude (numpy.ndarray): FFT magnitude spectrum
    """
    n = len(signal_data)

    fft_vals = np.fft.fft(signal_data)
    magnitude = np.abs(fft_vals)

    freqs = np.fft.fftfreq(n, d=1.0/fps)

    return freqs, magnitude
