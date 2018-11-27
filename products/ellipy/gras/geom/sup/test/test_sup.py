from ellipy.gras.la.la import *
from ellipy.gras.gen.gen import *
from ellipy.gras.geom.geom import *
from ellipy.gras.geom.tri.tri import *
from ellipy.gras.geom.sup.sup import *
import numpy as np
import pytest
from ellipy.elltool.conf.properties.Properties import Properties
import scipy.io
import scipy as sc
import os


class TestSup:

    def test_sup_2_boundary_2(self):
        q_mat = np.diag([1, 2])
        s_mat = orth_transl(np.array([1, 0]), np.array([1, 1]))
        q_mat = np.matmul(np.dot(s_mat.transpose(), q_mat), s_mat)
        dir_mat = circle_part(100)
        self.aux_test_sup_boundary(sup_2_boundary_2, sup_2_boundary_2, q_mat, dir_mat)

    def test_sup_2_boundary_3(self):
        q_mat = np.diag([1, 2, 3])
        s_mat = orth_transl(np.array([1 ,0, 0]), np.array([1, 1, 1]))
        q_mat = np.dot(np.dot(s_mat.transpose(), q_mat), s_mat)
        dir_mat, face_mat = sphere_tri(6)
        self.aux_test_sup_boundary(sup_2_boundary_3, sup_2_boundary_3, q_mat, dir_mat, face_mat)

    def aux_test_sup_boundary(self, f_boundary, f_check_boundary, q_mat, dir_mat, face_mat = []):
        max_norm = 1 + 1e-3
        min_norm = 1
        max_tol = 1e-14
        sup_vec = np.sqrt((sum((np.dot(dir_mat, q_mat)*dir_mat).transpose())))
        if f_boundary.__name__ == 'sup_2_boundary_3':
            x_mat = f_boundary( dir_mat, sup_vec, face_mat)
            x_exp_mat = f_check_boundary(dir_mat, sup_vec, face_mat)
        else:
            x_mat = f_boundary( dir_mat, sup_vec)
            x_exp_mat = f_check_boundary( dir_mat, sup_vec)
        real_tol = max(np.sqrt( sum(((x_mat - x_exp_mat)*(x_mat - x_exp_mat)).transpose())))
        assert (real_tol<=max_tol)
        y_mat = np.dot(x_mat, np.linalg.pinv(sc.linalg.sqrtm(q_mat)))
        n_vec = np.sqrt(sum((y_mat*y_mat).transpose()))
        assert ((max(n_vec) <= max_norm))
        assert ((min(n_vec) >= min_norm))
