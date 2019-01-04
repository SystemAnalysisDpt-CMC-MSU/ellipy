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

    def test_ell_reg_consitent(self):
        def check_apx_reg(f_apx_method: str):
            ell_arr = np.array([ell, ell.get_copy([ell]).flat[0]])
            apx_ell = getattr(ell, f_apx_method)(ell_arr, np.array([[1], [0], [0]]))
            ell_reg_shape_mat = apx_ell[0].get_shape_mat()
            max_diff = np.max(np.abs(0.25 * ell_reg_shape_mat[:] - sh_reg_mat[:]))
            is_ok = max_diff <= __CMP_TOL
            assert is_ok

        def check_is_pos(inp_mat: np.ndarray, is_exp_ok: bool, delta: float, *args):
            is_ok = is_mat_pos_def(inp_mat, abs_tol + delta, *args)
            assert is_ok == is_exp_ok
            if args != ():
                is_sem_pos_def = args[0]
            else:
                is_sem_pos_def = False
            if not is_sem_pos_def:
                ell_not_deg = self.ellipsoid(inp_mat, abs_tol=abs_tol + delta)
                is_ok = not ell_not_deg.is_degenerate([ell_not_deg])
                assert is_ok == is_exp_ok

        def master_check_is_pos(inp_mat: np.ndarray):
            eps_val = abs_tol * 1e-5
            check_is_pos(inp_mat, False, 0)
            check_is_pos(inp_mat, True, 0, True)
            check_is_pos(inp_mat, False, eps_val)
            check_is_pos(inp_mat, True, -eps_val)

        abs_tol = self.__ABS_TOL
        __CMP_TOL = 1e-12
        sh_mat = np.array([[4, 4, 14], [4, 4, 14], [14, 14, 78]])
        sh_reg_mat = reg_pos_def_mat(sh_mat, abs_tol)
        master_check_is_pos(sh_reg_mat)
        #
        ell = self.ellipsoid(sh_mat, abs_tol=abs_tol)
        assert np.array_equal(ell.get_shape_mat(), sh_mat)
        #
        check_apx_reg('minksum_ia')
        check_apx_reg('minksum_ea')
