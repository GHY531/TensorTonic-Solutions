import numpy as np

def finite_difference_derivative(coefficients: list, x: float, h: float) -> tuple[float, float, float]:
    """
    Returns the value at x, the value at x plus h, and the estimated slope.
    """
    length = len(coefficients)
    polynomial_value = 0
    for i in range (length):
        polynomial_value = float(polynomial_value + coefficients[i] * (x ** i))
    x_plus_h_value = 0
    for i in range (length):
        x_plus_h_value = float(x_plus_h_value + coefficients[i] * ((x + h) ** i))
    slope = float((x_plus_h_value - polynomial_value) / h)
    return (polynomial_value, x_plus_h_value, slope)