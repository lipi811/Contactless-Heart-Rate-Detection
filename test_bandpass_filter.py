import cv2
import numpy as np
from src.face_detection import detect_face
from src.bandpass_filter import temporal_bandpass_filter

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)

signal_buffer = []
buffer_size = 300   # ~10 seconds at 30 FPS

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_face(frame)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        roi = frame[y:y+h, x:x+w]

        # Extract green channel mean (rPPG signal)
        green_mean = np.mean(roi[:, :, 1])
        signal_buffer.append(green_mean)

        if len(signal_buffer) > buffer_size:
            signal_buffer.pop(0)

        if len(signal_buffer) == buffer_size:
            filtered = temporal_bandpass_filter(
                np.array(signal_buffer),
                fps
            )

            # Normalize for visualization
            norm = (filtered - np.min(filtered)) / (np.max(filtered) - np.min(filtered))
            norm = (norm * 255).astype('uint8')

            cv2.putText(
                frame,
                "Temporal Filtering Active",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

    cv2.imshow("STEP 6: Temporal Bandpass Filter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
