import numpy as np

def gradient_check_product_chain(
    a: float,
    b: float,
    c: float,
    f: float,
    h: float,
) -> tuple[float, list, list, float]:
    """
    Returns loss, analytic gradients, numerical gradients, and maximum error.
    """
    V = np.array([a, b, c, f, h], dtype=np.float64)

    # Caculate Loss
    e = V[0] * V[1] + V[2]
    L = e * V[3]

    # Caculate analytic gradients
    dL_df = e
    dL_da = V[1] * V[3]
    dL_db = V[0] * V[3]
    dL_dc = V[3]
    Ana_grad = np.array([dL_da, dL_db, dL_dc, dL_df], dtype=np.float64)
    ana_grad = Ana_grad.tolist()
    
    # Caculate numerical gradients
    V_h = np.array([a+h, b+h, c+h, f+h], dtype=np.float64)
    DL_DA = (V[3] * (V_h[0] * V[1] + V[2]) - L) / h
    DL_DB = (V[3] * (V[0] * V_h[1] + V[2]) - L) / h
    DL_DC = (V[3] * (V[0] * V[1] + V_h[2]) - L) / h
    DL_DF = (V_h[3] * (V[0] * V[1] + V[2]) - L) / h
    Num_grad = np.array([DL_DA, DL_DB, DL_DC, DL_DF], dtype=np.float64)
    num_grad = Num_grad.tolist()

    return float(L), ana_grad, num_grad, float(np.max(np.abs(Ana_grad - Num_grad)))
    