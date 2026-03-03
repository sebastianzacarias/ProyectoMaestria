import numpy as np

def compute_rmse(player_kp, reference_kp):
    return np.sqrt(np.mean((player_kp - reference_kp) ** 2))