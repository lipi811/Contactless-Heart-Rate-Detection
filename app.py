import streamlit as st
import cv2
import numpy as np
import time

from src.face_detection import detect_face
from src.bandpass_filter import temporal_bandpass_filter
from src.signal_amplification import amplify_signal
from src.fft_analysis import compute_fft
from src.heart_rate_estimation import estimate_heart_rate
from src.pos_method import extract_pos_signal
from src.roi_utils import extract_cheek_rois

# -------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Contactless Heart Rate", layout="centered")
st.title("❤️ Contactless Heart Rate Estimation")
st.write("High-accuracy real-time heart rate estimation")

# -------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------
START = st.sidebar.button("▶ Start Camera")
STOP = st.sidebar.button("⏹ Stop Camera")

# -------------------------------------------------
# PARAMETERS (OPTIMAL)
# -------------------------------------------------
BUFFER_SIZE = 300          # ~10 seconds
AMPLIFICATION_GAIN = 10
BPM_MIN = 45
BPM_MAX = 120
BPM_HISTORY_SIZE = 10

# -------------------------------------------------
# UI PLACEHOLDERS
# -------------------------------------------------
frame_placeholder = st.empty()
bpm_placeholder = st.empty()
status_placeholder = st.empty()

# -------------------------------------------------
# CAMERA LOOP
# -------------------------------------------------
if START:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    time.sleep(0.5)  # camera warm-up

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    signal_buffer = []
    bpm_history = []
    prev_bpm = None
    last_update_time = time.time()

    status_placeholder.success("Camera started")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or STOP:
            break

        faces = detect_face(frame)

        if len(faces) > 0:
            left_cheek, right_cheek = extract_cheek_rois(frame, faces[0])

            if left_cheek.size > 0 and right_cheek.size > 0:
                pos_left = extract_pos_signal(left_cheek)
                pos_right = extract_pos_signal(right_cheek)

                pos_value = (pos_left + pos_right) / 2
                signal_buffer.append(pos_value)

                if len(signal_buffer) > BUFFER_SIZE:
                    signal_buffer.pop(0)

                # Update BPM once per second
                if len(signal_buffer) == BUFFER_SIZE and time.time() - last_update_time > 1:
                    last_update_time = time.time()

                    signal_np = np.array(signal_buffer)
                    filtered = temporal_bandpass_filter(signal_np, fps)
                    amplified = amplify_signal(filtered, gain=AMPLIFICATION_GAIN)

                    freqs, mag = compute_fft(amplified, fps)
                    bpm = estimate_heart_rate(freqs, mag)

                    # Physiological + peak tracking constraint
                    if bpm is not None and BPM_MIN <= bpm <= BPM_MAX:
                        if prev_bpm is None or abs(bpm - prev_bpm) < 12:
                            bpm_history.append(bpm)
                            prev_bpm = bpm

                            if len(bpm_history) > BPM_HISTORY_SIZE:
                                bpm_history.pop(0)

            # Draw face box
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display stable BPM
        if len(bpm_history) > 0:
            stable_bpm = int(np.median(bpm_history))
            bpm_placeholder.markdown(
                f"## ❤️ Heart Rate: **{stable_bpm} BPM**"
            )
        else:
            bpm_placeholder.markdown("## ⏳ Measuring heart rate...")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        time.sleep(0.03)

    cap.release()
    status_placeholder.warning("Camera stopped")

# -------------------------------------------------
# INSTRUCTIONS
# -------------------------------------------------
st.markdown("---")
st.markdown("### ℹ️ Instructions")
st.markdown("""
- Ensure **good lighting**
- Sit still and face the camera
- Click **Start Camera**
- Wait **~10 seconds** for stable BPM
- Click **Stop Camera** to end
""")
