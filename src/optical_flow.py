import cv2
import numpy as np

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

def initialize_points(gray_roi, max_corners=100):
    return cv2.goodFeaturesToTrack(
        gray_roi,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=10
    )

def track_motion(prev_gray, curr_gray, prev_points):
    """
    Safely track optical flow with size check
    """

    # If image size mismatch, reset
    if prev_gray.shape != curr_gray.shape:
        return None, None

    curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        prev_points,
        None,
        **lk_params
    )

    good_prev = prev_points[status.flatten() == 1]
    good_curr = curr_points[status.flatten() == 1]

    return good_prev, good_curr
