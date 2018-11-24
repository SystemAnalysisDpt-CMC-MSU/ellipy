from typing import Tuple, Union
import numpy as np
import math
from ellipy.gen.common.common import throw_error
from ellipy.gras.geom.tri import spheretri

def circle_part(n_points: int, return_apart: bool = False,
                angle_range_vec: np.ndarray = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    if angle_range_vec is None:
        angle_range_vec = np.array([0., 2 * np.pi])
    v_phi = np.expand_dims(np.linspace(angle_range_vec[0], angle_range_vec[1], n_points, endpoint=False), 1)
    if return_apart:
        x_vec = np.cos(v_phi)
        y_vec = np.sin(v_phi)
        return x_vec, y_vec
    else:
        return np.concatenate([np.cos(v_phi), np.sin(v_phi)], 1)


def sphere_part(n_points: int) -> np.ndarray:
    

    def unique_directions(a_mat:np.ndarray, tol: float) -> np.ndarray:
        n_rows = aMat.shape[0]
        ind_remove_vec = np.zeros((nRows,1), dtype=bool)
        
        for i_row in range(0, n_rows-1):
            diff_mat = a_mat[-(n_rows - i_row - 1):, :] + np.tile(a_mat[i,:], ([n_rows - i - 1, 1]))
            diff_norm_vec = np.sqrt(np.sum(diff_mat*diff_mat, 1))
            if np.nonzero(diff_norm_vec < tol)[0].size > 0:
                if np.sum(a_mat[i_row,:]) < 0:
                    ind_remove_vec(i_row + np.nonzero(diff_norm_vec < tol)[0][0] + 1) = true
                else:
                    ind_remove_vec(i_row) = true
        return a_mat[ind_remove_vec,:];


    def sphere_distance(a_mat:np.ndarray, b_vec:np.ndarray) -> np.ndarray:
        dot_prod_vec = np.sum(a_mat * np.tile(b_vec, ([a_mat.shape[0], 1])), 1);#sum(bsxfun(@times,aMat,bVec), 2);
        return np.arccos(dot_prod_vec);
    

    if n_points <= 0:
        throw_error('wrongInput:n_points', 'n_points should be positive integer')
    
    if n_points < 21:
        depth = 1
    else:
        depth = math.ceil(log2((n_points - 1.) / 5) / 2)
        
    p_mat = unique_directions(sphere_tri(depth), 1e-8)

    x_dist_vec = sphere_distance(p_mat, np.array([1, 0, 0]))
    y_dist_vec = sphere_distance(p_mat, np.array([0, 1, 0]))
    z_dist_vec = sphere_distance(p_mat, np.array([0, 0, 1]))
    
    r_mat = np.zeros(shape=(n_points, 3))
    for i_point in range(0, n_points):
        mod_i = np.mod(i_point, 3)
        if mod_i == 1:
            i_min = np.argmin(x_dist_vec)
            x_dist_vec[i_min] = 2*pi
        elif mod_i == 2:
            i_min = np.argmin(y_dist_vec)
            y_dist_vec[i_min] = 2*pi
        else:
            i_min = np.argmin(z_dist_vec)
            z_dist_vec[i_min] = 2*pi
        r_mat[i_point] = p_mat[i_min]
    return r_mat

