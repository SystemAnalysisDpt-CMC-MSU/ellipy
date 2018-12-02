from ellipy.gen.common.common import throw_error, is_member
from typing import List, Dict, Any, Tuple
import numpy as np
import ellipy.elltool.conf.properties as p


class ParsePropMixin:
    __PROP_NAME_LIST = ['version', 'is_verbose', 'abs_tol', 'rel_tol', 'reg_tol',
                        'ode_solver_name', 'is_ode_norm_control', 'is_enabled_ode_solver_options',
                        'n_plot_2d_points', 'n_plot_3d_points', 'n_time_grid_points']
    __PROP_CHECK_FUNC_LIST = [
        lambda x: isinstance(x, str),  # version
        lambda x: isinstance(x, bool),  # isVerbose
        lambda x: isinstance(x, float) and x > 0,  # absTol
        lambda x: isinstance(x, float) and x > 0,  # relTol
        lambda x: isinstance(x, float) and x > 0,  # regTol
        lambda x: isinstance(x, str) and x in ['ode45', 'ode23', 'ode113'],  # ODESolverName
        lambda x: isinstance(x, bool),  # isODENormControl
        lambda x: isinstance(x, bool),  # isEnabledOdeSolverOptions
        lambda x: isinstance(x, int) or (isinstance(x, float) and x % 1 == 0.),  # nPlot2dPoints
        lambda x: isinstance(x, int) or (isinstance(x, float) and x % 1 == 0.),  # nPlot3dPoints
        lambda x: isinstance(x, int) or (isinstance(x, float) and x % 1 == 0.),  # nTimeGridPoints
    ]

    @staticmethod
    def parse_prop(args: Dict[str, Any], needed_prop_name_list: List[str] = None) -> Tuple[List[Any], Dict[str, Any]]:

        if needed_prop_name_list is None:
            needed_prop_name_list = ParsePropMixin.__PROP_NAME_LIST
            check_func_list = ParsePropMixin.__PROP_CHECK_FUNC_LIST
        else:
            is_there_vec, ind_there_vec = is_member(needed_prop_name_list, ParsePropMixin.__PROP_NAME_LIST)
            if ~np.all(is_there_vec):
                throw_error('wrongInput', 'properties {} are unknown'.format(
                    [pair[0] for pair in zip(needed_prop_name_list, np.logical_not(is_there_vec)) if pair[1]]))
            check_func_list = [ParsePropMixin.__PROP_CHECK_FUNC_LIST[ind] for ind in list(ind_there_vec)]

        pre_prop = p.Properties.Properties.get_prop_dict()

        prop_list = []

        args = args.copy()
        for i_prop in range(len(needed_prop_name_list)):
            prop_name = needed_prop_name_list[i_prop]
            check_func = check_func_list[i_prop]
            if prop_name in args:
                prop_val = args[prop_name]
                del args[prop_name]
            else:
                prop_val = pre_prop[prop_name]
            if not check_func(prop_val):
                throw_error('wrongInput', 'Property {} has wrong values'.format(prop_name))
            prop_list.append(prop_val)
        return prop_list, args
