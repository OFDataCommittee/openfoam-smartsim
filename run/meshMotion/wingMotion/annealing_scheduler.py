import numpy as np

def annealing_weight(epoch, T_start, T_end, sharpness=10):

    if epoch < T_start:
        return 0.0
    elif epoch > T_end:
        return 1.0
    else:
        # 标准化到 [0,1]
        x = (epoch - T_start) / (T_end - T_start)
        # S 型函数，中心点在 0.5
        return float(1 / (1 + np.exp(-sharpness * (x - 0.5))))