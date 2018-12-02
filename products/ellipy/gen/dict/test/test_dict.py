from ellipy.gen.dict.dict import *
import copy


class TestStructCompare:
    SXComp = dict()
    SYComp = dict()

    @classmethod
    def setup_class(cls):
        s = {'a': 1, 'b': 2}
        s_arr = np.array([s, s.copy()])
        s2_arr = copy.deepcopy(s_arr)
        s_arr[1]['b'] = 3
        x_arr = copy.deepcopy(s_arr)
        y_arr = copy.deepcopy(s2_arr)
        x_arr[0]['c'] = copy.deepcopy(s_arr)
        x_arr[1]['c'] = copy.deepcopy(s_arr)
        y_arr[0]['c'] = copy.deepcopy(s2_arr)
        y_arr[1]['c'] = copy.deepcopy(s2_arr)
        x_arr = np.array([x_arr, copy.deepcopy(x_arr)])
        y_arr = np.array([y_arr, copy.deepcopy(y_arr)])
        cls.SXComp = x_arr
        cls.SYComp = y_arr

    def test_simpledict_neg(self):
        s1 = {'a': 1, 'b': 2}
        s2 = {'a': 2, 'b': 2}
        is_eq, _ = dict_compare(s1, s2, 0.)
        assert not is_eq

    def test_vectorial_dict(self):
        def check(inp_s1, inp_s2):
            is_eq, _ = dict_compare(inp_s1, inp_s2, 0.)
            assert not is_eq

        s1 = np.array([[dict(), dict()], [dict(), dict()]])
        s1[1, 1]['alpha'] = 4
        s2 = np.array([dict(), dict(), dict(), dict()])
        s2[3]['alpha'] = 4
        check(s1, s2)
        check(s1, np.expand_dims(s2, 1))

        s1 = {'alpha': np.array([[dict(), dict(), dict()], [dict(), dict(), dict()]])}
        s1['alpha'][1, 2]['a'] = 6
        s2 = {'alpha': np.array([dict(), dict(), dict(), dict(), dict(), dict()])}
        s2['alpha'][5]['a'] = 6
        check(s1, s2)
        s2['alpha'] = np.expand_dims(s2['alpha'], 1)
        check(s1, s2)

    def test_simpledict_int64(self):
        def check(val1, val2, tol, exp_res):
            s1 = {'a': 1, 'b': val1}
            s2 = {'a': 1, 'b': val2}
            is_eq, _ = dict_compare(s1, s2, tol)
            assert is_eq == exp_res

        check(np.int64(1), np.int64(1), 0., True)
        check(np.uint64(1), np.uint64(2), 3., True)
        check(np.uint64(1), np.uint64(2), 0., False)

    def test_inf(self):
        s1 = {'a': 1, 'b': [np.nan, np.inf, -np.inf, 1]}
        is_eq, _ = dict_compare(s1, s1, 0.)
        assert is_eq
        s1['b'] = np.array(s1['b'])
        is_eq, _ = dict_compare(s1, s1, 0.)
        assert is_eq

    def test_simpledict_negative(self):
        s1 = {'a': 1, 'b': np.nan}
        s2 = {'a': 1, 'b': 2}
        is_eq, _ = dict_compare(s1, s2, 0.)
        assert not is_eq

    def test_simpledict_positive2(self):
        s1 = {'a': 1, 'b': 2}
        s2 = {'a': 1, 'b': 2}
        is_eq, _ = dict_compare(s1, s2, 0.)
        assert is_eq

    def test_simpledict2_negative(self):
        s1 = {'a': {'a': 1. + 1e-10, 'b': 1}, 'b': 2}
        s2 = {'a': {'a': 1., 'b': 1}, 'b': 2}
        is_eq, _ = dict_compare(s1, s2, 1e-11)
        assert not is_eq

    def test_simpledict2_positive(self):
        s1 = {'a': {'a': 1. + 1e-10, 'b': 1}, 'b': 2}
        s2 = {'a': {'a': 1., 'b': 1}, 'b': 2}
        is_eq, _ = dict_compare(s1, s2, 1e-9)
        assert is_eq

    def test_simpledict3_negative(self):
        s1 = {'a': {'a': np.nan, 'b': 1}, 'b': 2}
        s2 = {'a': {'a': 1., 'b': 1}, 'b': 2}
        is_eq, _ = dict_compare(s1, s2, 0.)
        assert not is_eq

    def test_simpledictarray1_negative(self):
        s1 = [
            {'a': {'a': 1. + 1e-10, 'b': 1}, 'b': 2},
            {'a': {'a': np.nan, 'b': 1}, 'b': 2}
        ]
        s2 = [
            {'a': {'a': 1., 'b': 1}, 'b': 2},
            {'a': {'a': 1., 'b': 1}, 'b': 2}
        ]
        is_eq, _ = dict_compare(s1, s2, 0.)
        assert not is_eq
        is_eq, _ = dict_compare(np.array(s1), np.array(s2), 0.)
        assert not is_eq

    def test_simpledictarray1_positive(self):
        s1 = [
            {'a': {'a': 1., 'b': 1}, 'b': 2},
            {'a': {'a': 1., 'b': 1}, 'b': 2}
        ]
        s2 = copy.deepcopy(s1)
        is_eq, _ = dict_compare(s1, s2, 0.)
        assert is_eq
        is_eq, _ = dict_compare(np.array(s1), np.array(s2), 0.)
        assert is_eq

    def test_complex1_positive(self):
        is_eq, report_str = dict_compare(self.SXComp, copy.deepcopy(self.SXComp), 0.)
        assert is_eq, report_str

    def test_complex1_negative(self):
        is_eq, report_str = dict_compare(self.SXComp, self.SYComp, 0.)
        assert not is_eq, report_str

    def test_optional_tolerance_arg(self):
        is_eq, report_str = dict_compare(self.SXComp, self.SYComp, 0.)
        is_eq2, report_str2 = dict_compare(self.SXComp, self.SYComp)
        assert is_eq == is_eq2
        assert report_str == report_str2

    def test_complex2_negative(self):
        s1 = {'a': 1, 'b': np.repeat(np.array([[2., np.nan, 3.]]), 2, 0)}
        s2 = {'a': 2, 'b': np.repeat(np.array([[1., np.nan, 2.]]), 2, 0)}
        is_eq, report_str = dict_compare(s1, s2, 0.1)
        assert not is_eq
        assert report_str.count('Max. ') == 2

    def test_differentsize_negative(self):
        s1 = {'a': 1, 'b': np.repeat(np.array([[2., np.nan, 3., 3.]]), 2, 0)}
        s2 = {'a': 2, 'b': np.repeat(np.array([[1., np.nan, 2.]]), 2, 0)}
        is_eq, report_str = dict_compare(s1, s2, 0.1)
        assert not is_eq
        assert report_str.count('Max. ') == 1
        assert report_str.count('Different sizes') == 1

    def test_list_positive(self):
        s1 = {'a': 1, 'b': [[np.nan], [{'c': ['aaa']}]]}
        is_eq, _ = dict_compare(s1, s1, 0.)
        assert is_eq

    def test_cell_negative(self):
        s1 = {'a': 1, 'b': [[np.nan], [{'c': ['aaa']}]]}
        s2 = {'a': 1, 'b': [[np.nan], [{'c': ['bbb']}]]}
        is_eq, report_str = dict_compare(s1, s2, 0.)
        assert not is_eq
        assert report_str.count('values are different') == 1

    def test_simpledict_order_positive(self):
        s1 = {'a': 1, 'b': 2}
        s2 = {'b': 2, 'a': 1}
        is_eq, _ = dict_compare(s1, s2, 0.)
        assert is_eq

    def test_relative_negative(self):
        def check_neg(rep_msg_count, inp_s1, inp_s2, *args):
            is_eq, report_str = dict_compare(inp_s1, inp_s2, *args)
            assert not is_eq
            assert report_str.count('Max. relative difference') == rep_msg_count

        s1 = {'a': 1e+10, 'b': 2e+12}
        s2 = {'b': 2e+12, 'a': 1e+10 + 1e+6}
        check_neg(1, s1, s2, 1e-10, 1e-5)

        s2 = {'b': 2e+12 - 1e+2, 'a': 1e+10 + 1e+6}
        check_neg(1, s1, s2, 1e+3, 1e-5)

        s2 = {'b': 2e+12 - 1e+9, 'a': 1e+10 + 1e+6}
        check_neg(2, s1, s2, 1e+3, 1e-5)

        s1 = {'a': 1e+6 - 2., 'b': 2e+6, 'c': 'aab'}
        s2 = {'a': 1e+6, 'b': 2e+6 + 4., 'c': 'aab'}
        check_neg(2, s1, s2, 1., 1e-7)

    def test_relative_positive(self):
        def check_pos(inp_s1, inp_s2, *args):
            is_eq, _ = dict_compare(inp_s1, inp_s2, *args)
            assert is_eq

        s1 = {'a': 1e+6 - 0.5, 'b': 2e+6, 'c': 'aab'}
        s2 = {'a': 1e+6, 'b': 2e+6 + 1., 'c': 'aab'}
        check_pos(s1, s2, 1e-10, 1e-6)

        s1 = {'a': 1e+10, 'b': 2e+12}
        s2 = {'b': 2e+12, 'a': 1e+10 + 1e+2}
        check_pos(s1, s2, 1e-10, 1e-5)

        s2 = {'b': 2e+12 - 1e+4, 'a': 1e+10 + 1e+2}
        check_pos(s1, s2, 1e+3, 1e-5)
