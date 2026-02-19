import numpy as np
from src.regression_cnn import build_regression_cnn

# -----------------------------
# DUMMY TRAINING DATA (FOR TESTING)
# -----------------------------
# Replace this later with real dataset (UBFC / PURE)

num_samples = 100
signal_length = 300

# Fake rPPG signals
X_train = np.random.rand(num_samples, signal_length, 1)

# Fake heart rates (60–100 BPM)
y_train = np.random.uniform(60, 100, num_samples)

# -----------------------------
# BUILD & TRAIN MODEL
# -----------------------------
model = build_regression_cnn(input_length=signal_length)

model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=8
)

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save("models/regression_cnn.h5")

print("✅ CNN model trained and saved!")
