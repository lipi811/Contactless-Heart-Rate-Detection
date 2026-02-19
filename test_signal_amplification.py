import cv2
import numpy as np
from src.face_detection import detect_face
from src.bandpass_filter import temporal_bandpass_filter
from src.signal_amplification import amplify_signal

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)

signal_buffer = []
buffer_size = 300

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_face(frame)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        roi = frame[y:y+h, x:x+w]

        green_mean = np.mean(roi[:, :, 1])
        signal_buffer.append(green_mean)

        if len(signal_buffer) > buffer_size:
            signal_buffer.pop(0)

        if len(signal_buffer) == buffer_size:
            filtered = temporal_bandpass_filter(
                np.array(signal_buffer), fps
            )

            amplified = amplify_signal(filtered, gain=10)

            cv2.putText(
                frame,
                "Signal Amplified",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )

    cv2.imshow("STEP 7: Signal Amplification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
