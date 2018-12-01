import numpy as np
import pytest
from ellipy.gras.sym.sym import is_dependent
from ellipy.gras.sym.sym import var_replace


class TestSym:
    @staticmethod
    def compare_m_mat_with_numerics(m_mat, cor_mat, res_mat):
        n = 3
        is_ok = False
        c = np.zeros(m_mat.shape)
        for i in range(len(m_mat)):
            for j in range(len(m_mat[i])):
                c[i][j] = isinstance(m_mat[i][j], float)
                if c[i][j]:
                    [first_num_str, first_frac_str] = res_mat[i][j].split('.')
                    [second_nun_str, second_frac_str] = cor_mat[i][j].split('.')
                    if first_num_str == second_nun_str:
                        if first_frac_str[0:n-1] == second_frac_str[0:n-1]:
                            is_ok = True
                        else:
                            is_ok = False
                    else:
                        is_ok = False
                else:
                    is_ok = True
        return is_ok

    def test_is_dependent(self):
        assert is_dependent(np.array([['cos(t)', 'sin(t)'], ['-sin(t)', 'cost(t)']], dtype='str'))
        assert not is_dependent(np.array([['cos(t)', 'sin(t)'], ['-sin(t)', 'cost(t)']], dtype='str'), is_discrete=True)
        assert is_dependent(np.array([['cos(k)', 'k'], ['-sin(k)', 'cos(k)']], dtype='str'), is_discrete=True)
        assert not is_dependent(np.array([['cos(k)', 'k'], ['-sin(k)', 'cos(k)']], dtype='str'))
        assert is_dependent(np.array([['4*k + 6', 5], [6, 10]], dtype='str'), is_discrete=True)
        assert is_dependent(np.array([['k', 5], [6, 10]], dtype='str'), is_discrete=True)
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
                          [1,   0,  -190]])
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
        m_mat = np.array([[150.00655674, 1.11,  2.54],
                          [-10.453, 30.01, 100.45],
                          [1.3242,   0.342,  -190.901]])
        from_var_name = 'att'
        to_var_name = '0.8-' + from_var_name
        res_mat = var_replace(m_mat, from_var_name, to_var_name)
        cor_mat = np.array([['150.007', '1.11', '2.54'],
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
                          ['1/(t+3)*2^t^t', 1.9]])
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
                          ['sin(t)', 1.9]])
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
                          ['sin(t)', 1.9]])
        from_var_name = 't'
        to_var_name = '0.81-' + from_var_name
        try:
            res_mat = var_replace(from_var_name, to_var_name)
        except TypeError:
            pass
        if 'res_mat' in locals():
            print('error')

    def test_var_replace_empty_mat(self):
        m_mat = np.array([[]])
        from_var_name = 't'
        to_var_name = '0.81-' + from_var_name
        with pytest.raises(Exception) as e:
            var_replace(m_mat, from_var_name, to_var_name)
        assert 'wrongInput: m_mat' in str(e.value)
