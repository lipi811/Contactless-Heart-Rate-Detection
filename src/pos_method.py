import numpy as np

def extract_pos_signal(roi):
    """
    POS-based rPPG signal extraction (illumination-robust)
    """
    R = np.mean(roi[:, :, 2])
    G = np.mean(roi[:, :, 1])
    B = np.mean(roi[:, :, 0])

    S1 = G - B
    S2 = G + B - 2 * R

    std_S1 = np.std(S1)
    std_S2 = np.std(S2) + 1e-6

    alpha = std_S1 / std_S2

    return S1 + alpha * S2
