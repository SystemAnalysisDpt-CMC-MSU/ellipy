from ellipy.gras.la.la import *
from ellipy.gras.geom.tri.tri import *
from ellipy.gras.geom.sup.sup import *
import numpy as np
import scipy as sc


class TestSup:
    def test_sup_2_boundary_2(self):
        q_mat = np.diag([1, 2])
        s_mat = orth_transl(np.array([1, 0]), np.array([1, 1]))
        q_mat = s_mat.T @ q_mat @ s_mat
        dir_mat = circle_part(100)
        self.aux_test_sup_boundary(sup_2_boundary_2, sup_2_boundary_2, q_mat, dir_mat)

    def test_sup_2_boundary_3(self):
        q_mat = np.diag([1, 2, 3])
        s_mat = orth_transl(np.array([1, 0, 0]), np.array([1, 1, 1]))
        q_mat = s_mat.T @ q_mat @ s_mat
        dir_mat, face_mat = sphere_tri(6)
        self.aux_test_sup_boundary(sup_2_boundary_3, sup_2_boundary_3, q_mat, dir_mat, face_mat)

    @staticmethod
    def aux_test_sup_boundary(f_boundary, f_check_boundary, q_mat, dir_mat, *args):
        __MAX_NORM = 1 + 1e-3
        __MIN_NORM = 1
        __MAX_TOL = 1e-14
        sup_vec = np.sqrt(np.sum((dir_mat @ q_mat) * dir_mat, 1).T)
        x_mat = f_boundary(dir_mat, sup_vec, *args)
        x_exp_mat = f_check_boundary(dir_mat, sup_vec, *args)
        real_tol = np.max(np.sqrt(np.sum(((x_mat - x_exp_mat) ** 2).T, 1)))
        assert real_tol <= __MAX_TOL
        y_mat = np.linalg.lstsq(sc.linalg.sqrtm(q_mat).T, x_mat.T, - 1)[0].T
        n_vec = np.sqrt(np.sum(y_mat ** 2, 1))
        assert max(n_vec) <= __MAX_NORM
        assert min(n_vec) >= __MIN_NORM
