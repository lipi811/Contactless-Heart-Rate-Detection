import cv2
import numpy as np
import os
import tensorflow as tf

from src.face_detection import detect_face
from src.bandpass_filter import temporal_bandpass_filter
from src.signal_amplification import amplify_signal
from src.fft_analysis import compute_fft
from src.heart_rate_estimation import estimate_heart_rate

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
USE_CNN = False        # Set True to enable CNN
MODEL_PATH = "models/regression_cnn.h5"
BUFFER_SIZE = 300     # ~10 seconds at 30 FPS
AMPLIFICATION_GAIN = 10

# -------------------------------------------------
# LOAD CNN MODEL (OPTIONAL)
# -------------------------------------------------
cnn_model = None
if USE_CNN and os.path.exists(MODEL_PATH):
    cnn_model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ CNN model loaded")
elif USE_CNN:
    print("⚠️ CNN model not found, running FFT only")

# -------------------------------------------------
# CAMERA SETUP
# -------------------------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 30  # fallback

print(f"📷 Camera FPS: {fps}")
print("🚀 Contactless Heart Rate System Started")

signal_buffer = []

# -------------------------------------------------
# MAIN LOOP
# -------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_face(frame)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        roi = frame[y:y+h, x:x+w]

        # ---------------- rPPG EXTRACTION ----------------
        green_mean = np.mean(roi[:, :, 1])
        signal_buffer.append(green_mean)

        if len(signal_buffer) > BUFFER_SIZE:
            signal_buffer.pop(0)

        # ---------------- SIGNAL PROCESSING ----------------
        if len(signal_buffer) == BUFFER_SIZE:
            signal_np = np.array(signal_buffer)

            # Step 6: Temporal Bandpass Filter
            filtered = temporal_bandpass_filter(signal_np, fps)

            # Step 7: Signal Amplification
            amplified = amplify_signal(filtered, gain=AMPLIFICATION_GAIN)

            # ---------------- FFT HEART RATE ----------------
            freqs, mag = compute_fft(amplified, fps)
            bpm_fft = estimate_heart_rate(freqs, mag)

            if bpm_fft is not None:
                cv2.putText(
                    frame,
                    f"FFT HR: {int(bpm_fft)} BPM",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            # ---------------- CNN HEART RATE ----------------
            if cnn_model is not None:
                cnn_input = amplified.reshape(1, BUFFER_SIZE, 1)
                bpm_cnn = cnn_model.predict(cnn_input, verbose=0)[0][0]

                cv2.putText(
                    frame,
                    f"CNN HR: {int(bpm_cnn)} BPM",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2
                )

        # Draw face bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Contactless Heart Rate Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------------------------
# CLEANUP
# -------------------------------------------------
cap.release()
cv2.destroyAllWindows()
