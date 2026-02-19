import cv2
import numpy as np
import tensorflow as tf

from src.face_detection import detect_face
from src.bandpass_filter import temporal_bandpass_filter
from src.signal_amplification import amplify_signal

# Load trained CNN model
model = tf.keras.models.load_model("models/regression_cnn.h5")

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

            cnn_input = amplified.reshape(1, buffer_size, 1)
            bpm = model.predict(cnn_input, verbose=0)[0][0]

            cv2.putText(
                frame,
                f"CNN Heart Rate: {int(bpm)} BPM",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 0, 0),
                3
            )

    cv2.imshow("STEP 10: CNN Heart Rate", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
