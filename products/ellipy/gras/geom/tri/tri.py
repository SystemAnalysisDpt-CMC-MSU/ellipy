from typing import Tuple, Dict, Callable
import numpy as np


def ell_tube_2_tri(n_e_points: int, n_points: int) -> np.ndarray:
    pass


def ell_tube_discr_tri(n_dim: int, m_dim: int) -> np.ndarray:
    pass


def icosahedron() -> Tuple[np.ndarray, np.ndarray]:
    pass


def map_face_2_edge(v_mat: np.ndarray, f_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pass


def shrink_face_tri(v_mat: np.ndarray, f_mat: np.ndarray,
                    max_edge_len: float, n_max_steps: float = np.inf,
                    f_vert_adjust_func: Callable[[np.ndarray], np.ndarray] = None) -> \
        Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    def deal(inp_arr: np.ndarray) -> np.ndarray:
        return inp_arr

    if f_vert_adjust_func is None:
        f_vert_adjust_func = deal


def sphere_tri(depth: int) -> Tuple[np.ndarray, np.ndarray]:
    pass


def sphere_tri_ext(n_dim: int, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    pass
