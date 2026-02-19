def extract_forehead_roi(frame, face):
    """
    Extract forehead ROI from detected face
    """
    x, y, w, h = face

    fh_y1 = y + int(0.15 * h)
    fh_y2 = y + int(0.35 * h)

    fh_x1 = x + int(0.25 * w)
    fh_x2 = x + int(0.75 * w)

    return frame[fh_y1:fh_y2, fh_x1:fh_x2]


def extract_cheek_rois(frame, face):
    """
    Extract left and right cheek ROIs (stronger pulse signal)
    """
    x, y, w, h = face

    # Vertical mid-face region
    y1 = y + int(0.45 * h)
    y2 = y + int(0.65 * h)

    # Left cheek
    lx1 = x + int(0.10 * w)
    lx2 = x + int(0.35 * w)

    # Right cheek
    rx1 = x + int(0.65 * w)
    rx2 = x + int(0.90 * w)

    left_cheek = frame[y1:y2, lx1:lx2]
    right_cheek = frame[y1:y2, rx1:rx2]

    return left_cheek, right_cheek
