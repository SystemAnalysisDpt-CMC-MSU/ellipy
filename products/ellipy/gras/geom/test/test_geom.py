from ellipy.gras.geom.geom import *
import numpy as np


class TestGeom:
    def test_circle_part(self):
        __CALC_PRECISION = 1e-14

        def test_for_n_points(n_points: int):
            x_mat_1 = circle_part(n_points)
            x_mat_2, y_vec_2 = circle_part(n_points, True)
            x_mat_3 = \
                circle_part(n_points, angle_range_vec=np.array([0., 3 * np.pi / 2]))
            x_mat_4, y_vec_4 = \
                circle_part(n_points, True, angle_range_vec=np.array([0., 3 * np.pi / 2]))
            norm_vec_1 = np.sqrt(np.sum(x_mat_1 * x_mat_1, axis=1).reshape(-1, 1))
            norm_vec_2 = np.sqrt((x_mat_2 * x_mat_2 + y_vec_2 * y_vec_2).reshape(-1, 1))
            norm_vec_3 = np.sqrt(np.sum(x_mat_3 * x_mat_3, axis=1).reshape(-1, 1))
            norm_vec_4 = np.sqrt((x_mat_4 * x_mat_4 + y_vec_4 * y_vec_4).reshape(-1, 1))
            assert x_mat_1.shape[0] == n_points and x_mat_1.shape[1] == 2
            assert x_mat_2.shape[0] == n_points and x_mat_2.shape[1] == 1
            assert y_vec_2.shape[0] == n_points and y_vec_2.shape[1] == 1
            assert x_mat_3.shape[0] == n_points and x_mat_3.shape[1] == 2
            assert x_mat_4.shape[0] == n_points and x_mat_4.shape[1] == 1
            assert y_vec_4.shape[0] == n_points and y_vec_4.shape[1] == 1
            assert np.all(np.abs(norm_vec_1 - 1) < __CALC_PRECISION)
            assert np.all(np.abs(norm_vec_2 - 1) < __CALC_PRECISION)
            assert np.all(np.abs(norm_vec_3 - 1) < __CALC_PRECISION)
            assert np.all(np.abs(norm_vec_4 - 1) < __CALC_PRECISION)

        for num_points_vec in np.arange(1, 51):
            test_for_n_points(num_points_vec)

    def test_sphere_part(self):
        __CALC_PRECISION = 1e-14
        
        num_points = np.array([1, 2, 3, 20, 21, 22, 41, 42, 43, 100])
        
        for i in range(num_points.shape[0]):
            p_mat = sphere_part(num_points[i])
            norm_vec = np.sqrt(np.sum(p_mat * p_mat, 1))
            
            assert p_mat.shape[0] == num_points[i]
            assert p_mat.shape[1] == 3
            assert all(np.abs(norm_vec - 1.) < __CALC_PRECISION)
