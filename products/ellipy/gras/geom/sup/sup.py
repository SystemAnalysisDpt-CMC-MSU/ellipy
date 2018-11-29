import numpy as np


def sup_2_boundary_2(dir_mat, sup_vec):
    n_dirs = np.shape(dir_mat)[0]
    x_bound_mat = np.zeros((2, n_dirs), dtype=np.float64)
    for i_dir in range(n_dirs - 1):
        x_bound_mat[:, i_dir] = np.linalg.lstsq(dir_mat[i_dir:i_dir + 2],
                                                sup_vec[i_dir:i_dir + 2], -1)[0]

    x_bound_mat[:, -1] = np.linalg.lstsq(np.vstack((dir_mat[n_dirs - 1, :], dir_mat[0, :])),
                                         np.hstack((sup_vec[n_dirs - 1], sup_vec[0])), -1)[0]
    x_bound_mat = x_bound_mat.T
    return x_bound_mat


def sup_2_boundary_3(dir_mat, sup_vec, face_mat):
    n_faces = face_mat.shape[0]
    x_bound_mat = np.zeros((3, n_faces), dtype=np.float64)

    for i_face in range(n_faces):
        x_bound_mat[:, i_face] = np.linalg.lstsq(dir_mat[face_mat[i_face, :], :],
                                                 sup_vec[face_mat[i_face, :]], -1)[0]
    x_bound_mat = x_bound_mat.T
    return x_bound_mat
