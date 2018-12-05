import numpy as np
import pytest
from ellipy.gras.sym.sym import is_dependent
from ellipy.gras.sym.sym import var_replace
from ellipy.gen.common.common import is_numeric


class TestSym:
    __N_COMPARE_DECIMAL_DIGITS = 15

    @staticmethod
    def compare_m_mat_with_numerics(m_mat, cor_mat, res_mat):
        def cmp_up_to_digits(first_num_str, second_num_str, n_cmp_digits):
            first_list = first_num_str.split('.')
            second_list = second_num_str.split('.')
            is_res_ok = first_list[0] == second_list[0] and len(first_list) == len(second_list)
            if is_res_ok and len(first_list) > 1:
                is_res_ok = first_list[1][0:n_cmp_digits - 1] == \
                            second_list[1][0:n_cmp_digits - 1]
            return is_res_ok

        is_num_mat = np.reshape([is_numeric(el) for el in list(m_mat.flatten())], m_mat.shape)
        if np.any(is_num_mat.flatten()):
            if not np.array_equal(cor_mat[~is_num_mat], res_mat[~is_num_mat]):
                return False
        if not np.all(is_num_mat.flatten()):
            if not np.all(np.array([cmp_up_to_digits(cor_el, res_el, TestSym.__N_COMPARE_DECIMAL_DIGITS)
                                    for cor_el, res_el in
                                    zip(list(cor_mat[is_num_mat]), list(res_mat[is_num_mat]))])):
                return False
        return True

    def test_is_dependent(self):
        assert is_dependent(np.array([['cos(t)', 'sin(t)'], ['-sin(t)', 'cos(t)']], dtype='str'))
        assert not is_dependent(np.array([['cos(t)', 'sin(t)'], ['-sin(t)', 'cost(t)']], dtype='str'), is_discrete=True)
        assert is_dependent(np.array([['cos(k)', 'k'], ['-sin(k)', 'cos(k)']], dtype='str'), is_discrete=True)
        assert not is_dependent(np.array([['cos(k)', 'k'], ['-sin(k)', 'cos(k)']], dtype='str'))
        assert is_dependent(np.array([['cos(t)', 't'], ['-sin(t)', 'cos(t)']], dtype='str'))
        assert not is_dependent(np.array([['cos(t)', 't'], ['-sin(t)', 'cos(t)']], dtype='str'), is_discrete=True)
        assert is_dependent(np.array([['4*k + 6', 5], [6, 10]], dtype='str'), is_discrete=True)
        assert is_dependent(np.array([['k', 5], [6, 10]], dtype='str'), is_discrete=True)
        assert is_dependent(np.array([['4*t + 6', 5], [6, 10]], dtype='str'))
        assert is_dependent(np.array([['t', 5], [6, 10]], dtype='str'))
        assert not is_dependent(np.array([[3, 5], [5.9, 'z']], dtype='str'))

    def test_var_replace_int(self):
        m_mat = np.array([['t+t^2+t^3+sin(t)', 't^(1/2)+t*t*17'],
                          ['att+t2', 't+temp^t'],
                          ['1/(t+3)*2^t^t', 't-t^t']])
        from_var_name = 't'
        to_var_name = '15 - ' + from_var_name
        res_mat = var_replace(m_mat, from_var_name, to_var_name)
        cor_mat = np.array([['(15 - t)+(15 - t)^2+(15 - t)^3+sin((15 - t))', '(15 - t)^(1/2)+(15 - t)*(15 - t)*17'],
                            ['att+t2', '(15 - t)+temp^(15 - t)'],
                            ['1/((15 - t)+3)*2^(15 - t)^(15 - t)', '(15 - t)-(15 - t)^(15 - t)']])
        assert np.array_equal(cor_mat, res_mat)

    def test_var_replace_real(self):
        m_mat = np.array([['t+t^2+t^3+sin(t)', 't^(1/2)+t*t*17'],
                          ['att+t2', 't+temp^t'],
                          ['1/(t+3)*2^t^t', 't-t^t']])
        from_var_name = 'att'
        to_var_name = '10.8 - ' + from_var_name
        res_mat = var_replace(m_mat, from_var_name, to_var_name)
        cor_mat = np.array([['t+t^2+t^3+sin(t)', 't^(1/2)+t*t*17'],
                            ['(10.8 - att)+t2', 't+temp^t'],
                            ['1/(t+3)*2^t^t', 't-t^t']])
        assert np.array_equal(cor_mat, res_mat)

    def test_var_replace_spaces(self):
        m_mat = np.array([['t+    t^2+t^3+sin(t)', 't^(1/2)+t*t*17'],
                          ['att+t2', 't+temp^t'],
                          ['1/(t+3)*2^t^t', 't-t^t']])
        from_var_name = 't'
        to_var_name = '15 - ' + from_var_name
        res_mat = var_replace(m_mat, from_var_name, to_var_name)
        cor_mat = np.array([['(15 - t)+(15 - t)^2+(15 - t)^3+sin((15 - t))', '(15 - t)^(1/2)+(15 - t)*(15 - t)*17'],
                            ['att+t2', '(15 - t)+temp^(15 - t)'],
                            ['1/((15 - t)+3)*2^(15 - t)^(15 - t)', '(15 - t)-(15 - t)^(15 - t)']])
        assert np.array_equal(cor_mat, res_mat)

    def test_var_replace_combo_of_vars(self):
        m_mat = np.array([['t+tt+   ttt', 't^(t/2)+t*tt-t'],
                          ['atttt+ttt2', 't+temp^t+0.1'],
                          ['1/(t+3)*2^tt^ts', 'st-t^t']])
        from_var_name = 't'
        to_var_name = '0.01-' + from_var_name
        res_mat = var_replace(m_mat, from_var_name, to_var_name)
        cor_mat = np.array([['(0.01-t)+tt+ttt', '(0.01-t)^((0.01-t)/2)+(0.01-t)*tt-(0.01-t)'],
                            ['atttt+ttt2', '(0.01-t)+temp^(0.01-t)+0.1'],
                            ['1/((0.01-t)+3)*2^tt^ts', 'st-(0.01-t)^(0.01-t)']])
        assert np.array_equal(cor_mat, res_mat)

    def test_var_replace_vec(self):
        m_mat = np.array(['tt+t^3+sin(cos(tt))', 't^(1/2)+t*t*17', 'tt'])
        from_var_name = 'tt'
        to_var_name = '0.8-' + from_var_name
        res_mat = var_replace(m_mat, from_var_name, to_var_name)
        cor_mat = np.array(['(0.8-tt)+t^3+sin(cos((0.8-tt)))', 't^(1/2)+t*t*17', '(0.8-tt)'])
        assert np.array_equal(cor_mat, res_mat)

    def test_var_replace_one_elem(self):
        m_mat = np.array([['tt']])
        from_var_name = 'tt'
        to_var_name = '0.8-' + from_var_name
        res_mat = var_replace(m_mat, from_var_name, to_var_name)
        cor_mat = np.array([['(0.8-tt)']])
        assert np.array_equal(cor_mat, res_mat)

    def test_var_replace_no_to_rep_var(self):
        m_mat = np.array([['t']])
        from_var_name = 'tt'
        to_var_name = '0.8-' + from_var_name
        res_mat = var_replace(m_mat, from_var_name, to_var_name)
        cor_mat = np.array([['t']])
        assert np.array_equal(cor_mat, res_mat)

    def test_var_replace_no_to_rep_fun(self):
        m_mat = np.array([['sqrt(tt)']])
        from_var_name = 't'
        to_var_name = '0.8-' + from_var_name
        res_mat = var_replace(m_mat, from_var_name, to_var_name)
        cor_mat = np.array([['sqrt(tt)']])
        assert np.array_equal(cor_mat, res_mat)

    def test_var_replace_no_to_rep_mat(self):
        m_mat = np.array([['exp(t)', 'sin(ttt)', '-t'],
                          ['tttt', 't_ttt', '-0.6*t'],
                          ['tan(exp(t))', 'exp(10)', 'cos(t-ttt)']])
        from_var_name = 'tt'
        to_var_name = '0.8-' + from_var_name
        res_mat = var_replace(m_mat, from_var_name, to_var_name)
        cor_mat = np.array([['exp(t)', 'sin(ttt)', '-t'],
                            ['tttt', 't_ttt', '-0.6*t'],
                            ['tan(exp(t))', 'exp(10)', 'cos(t-ttt)']])
        assert np.array_equal(cor_mat, res_mat)

    def test_var_replace_one_int_elem(self):  # если взять 150 не строку, а просто число, то не работает. почему..
        m_mat = np.array([[150]])
        from_var_name = 't'
        to_var_name = '0.8-' + from_var_name
        res_mat = var_replace(m_mat, from_var_name, to_var_name)
        cor_mat = np.array([['150']])
        assert np.array_equal(cor_mat, res_mat)

    def test_var_replace_int_mat(self):  # если взять 150 не строку, а просто число, то не работает. почему..
        m_mat = np.array([[150, 1,  2],
                          [-10, 30, 100],
                          [1,   0,  -190]], dtype=object)
        from_var_name = 'att'
        to_var_name = '0.8-' + from_var_name
        res_mat = var_replace(m_mat, from_var_name, to_var_name)
        cor_mat = np.array([['150', '1', '2'],
                            ['-10', '30', '100'],
                            ['1', '0',  '-190']])
        assert np.array_equal(cor_mat, res_mat)

    def test_var_replace_real_elem(self):  # если взять 150 не строку, а просто число, то не работает. почему..
        m_mat = np.array([[0.01]])
        from_var_name = 'att'
        to_var_name = '0.8-' + from_var_name
        res_mat = var_replace(m_mat, from_var_name, to_var_name)
        cor_mat = np.array([['0.01']])
        assert self.compare_m_mat_with_numerics(m_mat, cor_mat, res_mat)
        assert np.array_equal(cor_mat, res_mat)

    def test_var_replace_real_mat(self):
        m_mat = np.array([[15.00000000000000655674, 1.11,  2.54],
                          [-10.453, 30.01, 100.45],
                          [1.3240000000000002,   0.342,  -190.901]])
        from_var_name = 'att'
        to_var_name = '0.8-' + from_var_name
        res_mat = var_replace(m_mat, from_var_name, to_var_name)
        cor_mat = np.array([['15.000000000000007', '1.11', '2.54'],
                            ['-10.453', '30.01', '100.45'],
                            ['1.324', '0.342',  '-190.901']])
        assert self.compare_m_mat_with_numerics(m_mat, cor_mat, res_mat)
        assert np.array_equal(cor_mat, res_mat)

    def test_var_replace_int_mat_elem(self):
        m_mat = np.array([['t+t^2+t^3+sin(t)', 't^(1/2)+t*t*17'],
                          ['att+t2', 't+temp^t'],
                          ['1/(t+3)*2^t^t', 1]])
        from_var_name = 'att'
        to_var_name = '10.8 - ' + from_var_name
        res_mat = var_replace(m_mat, from_var_name, to_var_name)
        cor_mat = np.array([['t+t^2+t^3+sin(t)', 't^(1/2)+t*t*17'],
                            ['(10.8 - att)+t2', 't+temp^t'],
                            ['1/(t+3)*2^t^t', '1']])
        assert np.array_equal(cor_mat, res_mat)

    def test_var_replace_real_mat_elem(self):
        m_mat = np.array([['t+t^2+t^3+sin(t)', 't^(1/2)+t*t*17'],
                          ['att+t2', 't+temp^t'],
                          ['1/(t+3)*2^t^t', 1.9]], dtype=object)
        from_var_name = 'att'
        to_var_name = '10.8 - ' + from_var_name
        res_mat = var_replace(m_mat, from_var_name, to_var_name)
        cor_mat = np.array([['t+t^2+t^3+sin(t)', 't^(1/2)+t*t*17'],
                            ['(10.8 - att)+t2', 't+temp^t'],
                            ['1/(t+3)*2^t^t', '1.9']])
        assert self.compare_m_mat_with_numerics(m_mat, cor_mat, res_mat)
        assert np.array_equal(cor_mat, res_mat)

    def test_var_replace_mixed_mat(self):
        m_mat = np.array([['3.2',    't + 3.2'],
                          [9.8,      -34],
                          ['sin(t)', 1.9]], dtype=object)
        from_var_name = 't'
        to_var_name = '0.81-' + from_var_name
        res_mat = var_replace(m_mat, from_var_name, to_var_name)
        cor_mat = np.array([['3.2',    '(0.81-t)+3.2'],
                            ['9.8',      '-34'],
                            ['sin((0.81-t))', '1.9']])
        assert self.compare_m_mat_with_numerics(m_mat, cor_mat, res_mat)
        assert np.array_equal(cor_mat, res_mat)

    def test_var_replace_wrong_args(self):
        m_mat = np.array([['3.2',    't + 3.2'],
                          [9.8,      -34],
                          ['sin(t)', 1.9]], dtype=object)
        from_var_name = 't'
        with pytest.raises(Exception) as e:
            # noinspection PyArgumentList
            var_replace(m_mat, from_var_name)
            assert 'wrongInput:m_mat' in str(e.value)

    def test_var_replace_empty_mat(self):
        m_mat = np.array([[]])
        from_var_name = 't'
        to_var_name = '0.81-' + from_var_name
        with pytest.raises(Exception) as e:
            var_replace(m_mat, from_var_name, to_var_name)
            assert 'wrongInput:m_mat' in str(e.value)
