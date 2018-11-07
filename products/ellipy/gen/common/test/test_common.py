from ellipy.gen.common.common import *
import pytest
import warnings


def get_caller_name_function1():
    method_name, class_name = get_caller_name_ext(1)
    GetCallerNameExtTestClass.set_caller_info(method_name, class_name)


def get_caller_name_function2():
    def sub_function2():
        method_name, class_name = get_caller_name_ext(1)
        GetCallerNameExtTestClass.set_caller_info(method_name, class_name)

    sub_function2()


def get_caller_name_function3():
    get_caller_name_function2()


class GetCallerNameExtTestClass:
    __method_name = None
    __class_name = None
    
    def __init__(self):
        method_name, class_name = get_caller_name_ext(1)
        GetCallerNameExtTestClass.set_caller_info(method_name, class_name)

    @staticmethod
    def get_caller_info():
        return GetCallerNameExtTestClass.__method_name, GetCallerNameExtTestClass.__class_name

    @staticmethod
    def set_caller_info(method_name, class_name):
        GetCallerNameExtTestClass.__method_name = method_name
        GetCallerNameExtTestClass.__class_name = class_name

    def simple_method(self):
        method_name, class_name = get_caller_name_ext(1)
        GetCallerNameExtTestClass.set_caller_info(method_name, class_name)

    def sub_function_method(self):
        def sub_function():
            method_name, class_name = get_caller_name_ext(1)
            GetCallerNameExtTestClass.set_caller_info(method_name, class_name)

        sub_function()

    def sub_function_method1(self):
        get_caller_name_function1()

    def sub_function_method2(self):
        get_caller_name_function2()

    def sub_function_method3(self):
        get_caller_name_function3()


class TestCommon:
    def test_get_caller_name_ext(self):
        test_class_obj = GetCallerNameExtTestClass()
        method_name, class_name = test_class_obj.get_caller_info()
        assert method_name == '__init__'
        assert class_name == 'ellipy.gen.common.test.test_common.GetCallerNameExtTestClass'

        test_class_obj.simple_method()
        method_name, class_name = test_class_obj.get_caller_info()
        assert method_name == 'simple_method'
        assert class_name == 'ellipy.gen.common.test.test_common.GetCallerNameExtTestClass'

        test_class_obj.sub_function_method()
        method_name, class_name = test_class_obj.get_caller_info()
        assert method_name == 'sub_function'
        assert class_name == 'ellipy.gen.common.test.test_common'

        test_class_obj.sub_function_method1()
        method_name, class_name = test_class_obj.get_caller_info()
        assert method_name == 'get_caller_name_function1'
        assert class_name == 'ellipy.gen.common.test.test_common'

        test_class_obj.sub_function_method2()
        method_name, class_name = test_class_obj.get_caller_info()
        assert method_name == 'sub_function2'
        assert class_name == 'ellipy.gen.common.test.test_common'

        test_class_obj.sub_function_method3()
        assert method_name == 'sub_function2'
        assert class_name == 'ellipy.gen.common.test.test_common'
        
        get_caller_name_function1()
        method_name, class_name = test_class_obj.get_caller_info()
        assert method_name == 'get_caller_name_function1'
        assert class_name == 'ellipy.gen.common.test.test_common'

        get_caller_name_function2()
        method_name, class_name = test_class_obj.get_caller_info()
        assert method_name == 'sub_function2'
        assert class_name == 'ellipy.gen.common.test.test_common'

        get_caller_name_function3()
        method_name, class_name = test_class_obj.get_caller_info()
        assert method_name == 'sub_function2'
        assert class_name == 'ellipy.gen.common.test.test_common'

    def test_throw_error(self):
        def check(tag, message):
            exp_e = throw_error(tag, message, throw=False)
            with pytest.raises(ValueError) as res_e:  
                throw_error(tag, message)
            assert res_e.value.args == exp_e.args

        check('wrongInput', 'test message')
        check('wrongInput', 'test \\ message C:\\SomeFolder\\sdf/sdf/sdfsdf')

    def test_throw_warn(self):
        def check(tag, message):
            id_str = 'ELLIPY:GEN:COMMON:TEST:TEST_COMMON:CHECK:' + tag
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                throw_warn(tag, message)
                assert len(w) == 1
                assert issubclass(w[-1].category, RuntimeWarning)
                assert str(w[-1].message) == id_str + ':' + message

        check('wrongInput', 'test message')
        check('wrongInput', 'test \n message C:\\SomeFolder\\sdf/sdf/sdfsdf')

    def test_is_numeric(self):
        __NUMERIC = [True, 1, -1, 1.0, 1+1j]
        __NOT_NUMERIC = [object(), 'string', u'unicode', None]
        for x in __NUMERIC:
            for y in (x, [x], [x] * 2):
                for z in (y, np.array(y)):
                    assert is_numeric(z)
        for x in __NOT_NUMERIC:
            for y in (x, [x], [x] * 2):
                for z in (y, np.array(y)):
                    assert not is_numeric(z)
        for kind, dtypes in np.sctypes.items():
            if kind != 'others':
                for dtype in dtypes:
                    assert is_numeric(np.array([0], dtype=dtype))

    def test_abs_rel_compare(self):
        def calc(*args):
            comp_is_eq, comp_abs_diff, comp_is_rel, \
                comp_rel_diff, comp_rel_mdiff, _ = abs_rel_compare(*args)
            comp_res = {
                'is_equal': comp_is_eq,
                'abs_diff': comp_abs_diff,
                'is_rel': comp_is_rel,
                'rel_diff': comp_rel_diff,
                'rel_mdiff': comp_rel_mdiff
            }
            return comp_res
  
        def check(dict1, dict2):
            assert dict1.keys() == dict2.keys()
            for key in dict1.keys():
                val1 = dict1[key]
                val2 = dict2[key]
                if isinstance(val1, np.ndarray):
                    assert np.array_equal(val1, val2)
                else:
                    assert val1 == val2

        # size error
        with pytest.raises(Exception) as e:  
            abs_rel_compare(np.array([1, 1]), np.array([[1], [1]]), 0.1, None, np.abs)
        assert 'wrongInput:wrongArgs' in str(e.value)
        # absTol error #1
        with pytest.raises(Exception) as e:  
            abs_rel_compare(np.array([1, 1]), np.array([1, 1]), -0.1, None, np.abs)
        assert 'wrongInput:wrongAbsTol' in str(e.value)
        # absTol error #2
        with pytest.raises(Exception) as e:  
            abs_rel_compare(np.array([1, 1]), np.array([1, 1]), np.array([0.1, 0.1]), None, np.abs)
        assert 'wrongInput:wrongAbsTol' in str(e.value)
        # absTol error #3
        with pytest.raises(Exception) as e:  
            abs_rel_compare(np.array([1, 1]), np.array([1, 1]), None, None, np.abs)
        assert 'wrongInput:wrongAbsTol' in str(e.value)
        # absTol error #4
        with pytest.raises(Exception) as e:  
            abs_rel_compare(np.array([1, 1]), np.array([1, 1]), np.array([]), None, np.abs)
        assert 'wrongInput:wrongAbsTol' in str(e.value)
        # relTol error #1
        with pytest.raises(Exception) as e:  
            abs_rel_compare(np.array([1, 1]), np.array([1, 1]), 0.1, -0.1, np.abs)
        assert 'wrongInput:wrongRelTol' in str(e.value)
        # relTol error #2
        with pytest.raises(Exception) as e:  
            abs_rel_compare(np.array([1, 1]), np.array([1, 1]), 0.1, np.array([0.1, 0.1]), np.abs)
        assert 'wrongInput:wrongRelTol' in str(e.value)
        # relTol error #3
        with pytest.raises(Exception) as e:  
            abs_rel_compare(np.array([1, 1]), np.array([1, 1]), 0.1, np.array([]), np.abs)
        assert 'wrongInput:wrongRelTol' in str(e.value)
        # fNormOp error
        with pytest.raises(Exception) as e:  
            abs_rel_compare(np.array([1, 1]), np.array([1, 1]), 0.1, None, 100)
        assert 'wrongInput:wrongNormOp' in str(e.value)

        # result tests
        res = calc(np.zeros((0,)), np.zeros((0,)), 0.5, None, np.abs)
        exp_res = {
            'is_equal': True, 'abs_diff': np.array([]), 'is_rel': False,
            'rel_diff': np.array([]), 'rel_mdiff': np.array([])}
        check(exp_res, res)

        x_vec = np.array([1, 2])
        y_vec = np.array([2, 4])
        res = calc(x_vec, y_vec, 2, None, np.abs)
        exp_res['is_equal'] = True
        exp_res['abs_diff'] = 2
        check(exp_res, res)

        res = calc(x_vec, y_vec, 1, None, np.abs)
        exp_res['is_equal'] = False
        check(exp_res, res)

        res = calc(x_vec, y_vec, 2, 2/3, np.abs)
        exp_res['is_equal'] = True
        check(exp_res, res)

        res = calc(x_vec, y_vec, 1., 2/3, np.abs)
        exp_res['is_rel'] = True
        exp_res['rel_diff'] = 2/3
        exp_res['rel_mdiff'] = 2
        check(exp_res, res)

        res = calc(x_vec, y_vec, 1, 0.5, np.abs)
        exp_res['is_equal'] = False
        check(exp_res, res)

        res = calc(x_vec, y_vec, 0.5, 0.5, np.abs)
        check(exp_res, res)

    def test_is_member_for_str(self):
        a_vec = ['asdfsdf', 'sdfsfd', 'sdfsdf', 'sdf']
        b_vec = ['sdf', 'sdfsdf', 'ssdfsfsdfsd', 'sdf']
        is_to_vec, ind_lo_vec = is_member(a_vec, b_vec)
        assert np.array_equal(np.array([False, False, True, True]), is_to_vec)
        assert np.array_equal(np.array([1, 0]), ind_lo_vec)

        is_to_vec, ind_lo_vec = is_member(a_vec, 'sdfsfd')
        assert np.array_equal(np.array([False, True, False, False]), is_to_vec)
        assert np.array_equal(np.array([0]), ind_lo_vec)

        is_to_vec, ind_lo_vec = is_member('sdfsfd', a_vec)
        assert np.array_equal(np.array(True), is_to_vec)
        assert np.array_equal(np.array([1]), ind_lo_vec)
        is_to_vec, ind_lo_vec = is_member('sdfsfd', 'sdfsfd')
        assert np.array_equal(np.array(True), is_to_vec)
        assert np.array_equal(np.array([0]), ind_lo_vec)
        is_to_vec, ind_lo_vec = is_member('sdfsfd', 'sdfsf')
        assert np.array_equal(np.array(False), is_to_vec)
        assert np.array_equal(np.array([]), ind_lo_vec)

        is_to_vec, ind_lo_vec = is_member('alpha', np.array(['a', 'b', 'c']))
        assert np.array_equal(np.array(False), is_to_vec)
        assert np.array_equal(np.array([]), ind_lo_vec)

        is_to_vec, ind_lo_vec = is_member(np.array(['a', 'b', 'c']), 'alpha')
        assert np.array_equal(np.array([False, False, False]), is_to_vec)
        assert np.array_equal(np.array([]), ind_lo_vec)
