import numpy as np
from scipy.signal import butter, filtfilt

def temporal_bandpass_filter(signal, fps):
    """
    Bandpass filter tuned for realistic heart rate range
    """
    lowcut = 0.9    # 54 BPM
    highcut = 2.5   # 150 BPM

    nyquist = 0.5 * fps
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(3, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal
