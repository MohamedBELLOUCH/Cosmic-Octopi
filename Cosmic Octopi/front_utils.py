import numpy as np
from typing import List, Union, Tuple


def length_scalarization(y: np.ndarray, ref_point: np.ndarray, lbda: np.ndarray) -> float:
    return float(np.min(np.maximum(y - ref_point, 0.0) / lbda))


def expected_pareto_front(
        Data,
        ref_point,
        n_points: int = 200,
        safety_epsilon: float = 1e-3
    ):
    Data = [np.asarray(A, dtype=float) for A in Data]
    ref_point = np.asarray(ref_point, dtype=float)
    if any(A.shape[1] != 2 for A in Data) or ref_point.shape[0] != 2:
        raise ValueError("This implementation is for the 2-dimensional case (M=2).")
    all_points = np.vstack(Data)
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    ranges = max_vals - min_vals
    Data_norm = [(A - min_vals) / ranges for A in Data]
    ref_norm = (ref_point - min_vals) / ranges
    thetas = np.linspace(safety_epsilon, np.pi/2 - safety_epsilon, n_points)
    P_norm = np.empty((n_points, 2))
    for i, theta in enumerate(thetas):
        lbda = np.array([np.cos(theta), np.sin(theta)])
        max_lengths = [max(length_scalarization(a, ref_norm, lbda) for a in A) for A in Data_norm]
        expected_length = np.mean(max_lengths)
        P_norm[i] = ref_norm + expected_length * lbda
    P = min_vals + P_norm * ranges
    return P


def interpolated_weak_pareto_front(
        A: Union[List[List[float]], np.ndarray],
        ref_point: Union[List[float], np.ndarray],
        n_points: int = 200,
        safety_epsilon: float = 1e-3
    ) -> np.ndarray:
    
    A = np.asarray(A, dtype=float)
    ref_point = np.asarray(ref_point, dtype=float)
    if A.shape[1] != 2 or ref_point.shape[0] != 2:
        raise ValueError("This implementation is for the 2-dimensional case (M=2).")
    min_vals = np.min(A, axis=0)
    max_vals = np.max(A, axis=0)
    ranges = max_vals - min_vals
    A_norm = (A - min_vals) / ranges
    ref_norm = (ref_point - min_vals) / ranges
    thetas = np.linspace(safety_epsilon, np.pi/2 - safety_epsilon, n_points)
    P_norm = np.empty((n_points, 2))
    for i, theta in enumerate(thetas):
        lbda = np.array([np.cos(theta), np.sin(theta)])
        projected_lengths = np.array([
            length_scalarization(a, ref_norm, lbda) for a in A_norm
        ])
        max_length = np.max(projected_lengths)
        P_norm[i] = ref_norm + max_length * lbda
    P = min_vals + P_norm * ranges
    return P


def RTS_pareto_front(
        Data,
        ref_point,
        n_points: int = 200,
        safety_epsilon: float = 1e-3
    ):
    Data = [np.asarray(A, dtype=float) for A in Data]
    ref_point = np.asarray(ref_point, dtype=float)
    if any(A.shape[1] != 2 for A in Data) or ref_point.shape[0] != 2:
        raise ValueError("This implementation is for the 2-dimensional case (M=2).")
    all_points = np.vstack(Data)
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    ranges = max_vals - min_vals
    Data_norm = [(A - min_vals) / ranges for A in Data]
    ref_norm = (ref_point - min_vals) / ranges
    thetas = np.linspace(safety_epsilon, np.pi/2 - safety_epsilon, n_points)
    P_norm = np.empty((n_points, 2))
    mean_vectors = np.array([np.mean(A, axis=0) for A in Data_norm])
    for i, theta in enumerate(thetas):
        lbda = np.array([np.cos(theta), np.sin(theta)])
        projected_lengths = np.array([length_scalarization(a, ref_norm, lbda) for a in mean_vectors])
        max_length = np.max(projected_lengths)
        P_norm[i] = ref_norm + max_length * lbda
    P = min_vals + P_norm * ranges
    return P

    

def STR_pareto_front(
        Data,
        ref_point,
        n_points: int = 200,
        safety_epsilon: float = 1e-3
    ):
    Data = [np.asarray(A, dtype=float) for A in Data]
    ref_point = np.asarray(ref_point, dtype=float)
    if any(A.shape[1] != 2 for A in Data) or ref_point.shape[0] != 2:
        raise ValueError("This implementation is for the 2-dimensional case (M=2).")
    all_points = np.vstack(Data)
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    ranges = max_vals - min_vals
    Data_norm = [(A - min_vals) / ranges for A in Data]
    ref_norm = (ref_point - min_vals) / ranges
    thetas = np.linspace(safety_epsilon, np.pi/2 - safety_epsilon, n_points)
    P_norm = np.empty((n_points, 2))
    for i, theta in enumerate(thetas):
        lbda = np.array([np.cos(theta), np.sin(theta)])
        expected_scalarizations = [
            np.mean([length_scalarization(a, ref_norm, lbda) for a in A])
            for A in Data_norm
        ]
        max_expected = max(expected_scalarizations)
        P_norm[i] = ref_norm + max_expected * lbda
    P = min_vals + P_norm * ranges
    return P