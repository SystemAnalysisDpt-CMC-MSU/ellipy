from ellipy.elltool.core.ellipsoid.Ellipsoid import *
from ellipy.gras.la.la import is_mat_pos_def, reg_pos_def_mat


class TestEllAuxTestCase:
    __ABS_TOL = 1e-8

    # noinspection PyMethodMayBeStatic
    def ellipsoid(self, *args, **kwargs):
        return Ellipsoid(*args, **kwargs)

    def test_ell_regularize(self):
        sh_mat = np.array([[4, 4, 14], [4, 4, 14], [14, 14, 78]], dtype=np.float64)
        is_ok = is_mat_pos_def(sh_mat, self.__ABS_TOL)
        assert not is_ok
        sh_mat = reg_pos_def_mat(sh_mat, self.__ABS_TOL)
        is_ok = is_mat_pos_def(sh_mat, self.__ABS_TOL, True)
        assert is_ok
