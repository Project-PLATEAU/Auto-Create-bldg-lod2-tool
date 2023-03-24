import numpy as np
import numpy.typing as npt
import math


def get_angle_degree_3d(v1: npt.NDArray[np.float_], v2: npt.NDArray[np.float_]) -> float:
    """2つの3次元ベクトルのなす角を求める

    Args:
        v1(NDArray[np.float_]): 3次元ベクトル
        v2(NDArray[np.float_]): 3次元ベクトル

    Returns:
        float: なす角(degree)
    """
    return np.rad2deg(math.acos(
        min(1, max(-1, np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)))
    ))
