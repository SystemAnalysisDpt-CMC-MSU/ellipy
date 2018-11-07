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
        args = {'absTol': test_abs_tol, 'relTol': test_rel_tol,
                'nPlot2dPoints': n_plot_2d_points, 'someOtherArg': some_arg}
        needed_prop = ['absTol', 'relTol']
        prop_val_list, _ = Properties.parse_prop(args, needed_prop)
        assert len(prop_val_list) == 2
        assert prop_val_list[0] == test_abs_tol and prop_val_list[1] == test_rel_tol
        # Negative test
        args['absTol'] = -test_abs_tol
        with pytest.raises(Exception) as e:  
            Properties.parse_prop(args, needed_prop)
        assert 'wrongInput' in str(e.value)
        args['absTol'] = test_abs_tol
        needed_prop[1] = 'notAProperty'
        with pytest.raises(Exception) as e:  
            Properties.parse_prop(args, needed_prop)
        assert 'wrongInput' in str(e.value)
