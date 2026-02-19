import tensorflow as tf

def build_regression_cnn(input_length=300):
    """
    Build 1D CNN model for heart rate regression.

    Parameters:
        input_length (int): Length of rPPG signal

    Returns:
        model (tf.keras.Model)
    """

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=5,
            activation='relu',
            input_shape=(input_length, 1)
        ),
        tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu'),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)   # Regression output (BPM)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )

    return model
