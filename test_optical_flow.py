import cv2
import numpy as np
from src.face_detection import detect_face
from src.optical_flow import initialize_points, track_motion

cap = cv2.VideoCapture(0)

prev_gray = None
prev_points = None
prev_shape = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_face(frame)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        roi_gray = gray[y:y+h, x:x+w]

        # Initialize or reinitialize feature points
        if prev_gray is None or roi_gray.shape != prev_shape:
            prev_gray = roi_gray.copy()
            prev_points = initialize_points(roi_gray)
            prev_shape = roi_gray.shape
        else:
            good_prev, good_curr = track_motion(
                prev_gray, roi_gray, prev_points
            )

            if good_curr is not None:
                for pt in good_curr:
                    px, py = pt.ravel()   # ✅ FIX HERE
                    cv2.circle(
                        frame,
                        (int(px + x), int(py + y)),
                        2,
                        (0, 0, 255),
                        -1
                    )

                prev_gray = roi_gray.copy()
                prev_points = good_curr.reshape(-1, 1, 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("STEP 2: Optical Flow (Stable)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
