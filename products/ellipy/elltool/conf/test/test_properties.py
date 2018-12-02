import pytest
from ellipy.elltool.conf.properties.Properties import *


class TestProperties:
    @classmethod
    def setup_class(cls):
        Properties.init()

    def test_parse_prop(self):
        # Positive test
        test_abs_tol = 1.
        test_rel_tol = 2.
        n_plot_2d_points = 3
        some_arg = 4.
        args = {'abs_tol': test_abs_tol, 'rel_tol': test_rel_tol,
                'n_plot_2d_points': n_plot_2d_points, 'some_other_arg': some_arg}
        needed_prop = ['abs_tol', 'rel_tol']
        prop_val_list, _ = Properties.parse_prop(args, needed_prop)
        assert len(prop_val_list) == 2
        assert prop_val_list[0] == test_abs_tol and prop_val_list[1] == test_rel_tol
        # Negative test
        args['abs_tol'] = -test_abs_tol
        with pytest.raises(Exception) as e:  
            Properties.parse_prop(args, needed_prop)
        assert 'wrongInput' in str(e.value)
        args['abs_tol'] = test_abs_tol
        needed_prop[1] = 'not_a_property'
        with pytest.raises(Exception) as e:  
            Properties.parse_prop(args, needed_prop)
        assert 'wrongInput' in str(e.value)
