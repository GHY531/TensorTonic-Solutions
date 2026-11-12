import numpy as np

def gradient_descent_step(values: list, gradients: list, learning_rate: float) -> tuple[list, float]:
    """
    Returns a fresh list of updated values and the predicted objective change.
    """
    values = np.array(values, dtype=np.float64)
    gradients = np.array(gradients, dtype=np.float64)
    learning_rate = np.float64(learning_rate)

    uploaded_values = values - gradients * learning_rate
    L_pred = np.sum(gradients * (uploaded_values - values))

    return uploaded_values.tolist(), float(L_pred)