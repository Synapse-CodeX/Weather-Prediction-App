import numpy as np

def apply_physical_constraints(prediction: np.ndarray, hour: int) -> np.ndarray:
    
    pred = prediction.copy()
    
    # Radiation is zero at night
    if hour < 6 or hour > 18:
        pred[1] = 0
    
    # Bounded values
    pred[2] = np.clip(pred[2], 0, 100)   # Cloud coverage
    pred[3] = max(0, pred[3])             # Rain (non-negative)
    pred[4] = np.clip(pred[4], 0, 100)    # Humidity
    pred[5] = max(0, pred[5])              # Wind speed
    
    return pred