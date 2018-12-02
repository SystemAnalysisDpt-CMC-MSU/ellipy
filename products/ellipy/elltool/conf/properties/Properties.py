from ellipy.gen.common.common import throw_error
from typing import List, Dict, Any
# noinspection PyProtectedMember
from ellipy.elltool.conf.properties._ParseProp import ParsePropMixin


class Properties(ParsePropMixin):
    # PROPERTIES - a static class, providing emulation of static properties for
    #             toolbox.
    #

    __DEFAULT_CONF_NAME = 'default'
    __SETUP_CLASS_NAME_LIST = []
    __conf_repo_mgr = None
    __implicit_init_call = False

    @staticmethod
    def _implicit_init():
        if not Properties.__implicit_init_call:
            Properties.init()
            Properties.__implicit_init_call = True

    @staticmethod
    def __get_basic_prop_list() -> List[float]:
        return [Properties.get_abs_tol(), Properties.get_rel_tol(), Properties.get_is_verbose()]

    @staticmethod
    def check_settings() -> None:
        inp_arg_list = Properties.__get_basic_prop_list()
        for setup_class_name in Properties.__SETUP_CLASS_NAME_LIST:
            obj = globals()[setup_class_name]()
            obj.check_settings(*inp_arg_list)

    @staticmethod
    def init() -> None:
        # confRepoMgr=elltool.conf.ConfRepoMgr();
        # confRepoMgr.selectConf(Properties.DEFAULT_CONF_NAME);
        # Properties.setConfRepoMgr(confRepoMgr);

        # TODO: change dictionary on real conf repo manager
        Properties.__conf_repo_mgr = {
            'version': '2.1',
            'is_verbose': False,
            'abs_tol': 1e-06,
            'rel_tol': 1e-05,
            'n_time_grid_points': 250,
            'ode_solver_name': 'ode45',
            'is_ode_norm_control': 'on',
            'is_enabled_ode_solver_options': False,
            'n_plot_2d_points': 200,
            'n_plot_3d_points': 200,
            'reg_tol': 1e-05
        }

        inp_arg_list = Properties.__get_basic_prop_list()
        for setup_class_name in Properties.__SETUP_CLASS_NAME_LIST:
            obj = globals()[setup_class_name]()
            obj.full_setup(*inp_arg_list)
        # elltool.logging.Log4jConfigurator.configure(confRepoMgr,...
        #     'islockafterconfigure',true);

    @staticmethod
    def get_conf_repo_mgr():
        if Properties.__conf_repo_mgr is None:
            Properties.init()
            if Properties.__conf_repo_mgr is None:
                throw_error('noConfRepoMgr', 'cannot initialize Configuration Repo Manager')
        return Properties.__conf_repo_mgr

    @staticmethod
    def set_conf_repo_mgr(conf_repo_mgr):
        Properties.__conf_repo_mgr = conf_repo_mgr

    # Public getters
    @staticmethod
    def get_version() -> str:
        return Properties.__get_option('version')

    @staticmethod
    def get_is_verbose() -> bool:
        return Properties.__get_option('is_verbose')

    @staticmethod
    def get_abs_tol() -> float:
        return Properties.__get_option('abs_tol')

    @staticmethod
    def get_rel_tol() -> float:
        return Properties.__get_option('rel_tol')

    @staticmethod
    def get_reg_tol() -> float:
        return Properties.__get_option('reg_tol')

    @staticmethod
    def get_n_time_grid_points() -> int:
        return Properties.__get_option('n_time_grid_points')

    @staticmethod
    def get_ode_solver_name() -> str:
        return Properties.__get_option('ode_solver_name')

    @staticmethod
    def get_is_ode_norm_control() -> bool:
        return Properties.__get_option('is_ode_norm_control')

    @staticmethod
    def get_is_enabled_ode_solver_options() -> bool:
        return Properties.__get_option('is_enabled_ode_solver_options')

    @staticmethod
    def get_n_plot_2d_points() -> int:
        return Properties.__get_option('n_plot_2d_points')

    @staticmethod
    def get_n_plot_3d_points() -> int:
        return Properties.__get_option('n_plot_3d_points')

    # Public setters
    @staticmethod
    def set_is_verbose(is_verb: bool) -> None:
        Properties.__set_option('is_verbose', is_verb)

    @staticmethod
    def set_n_plot_2d_points(n_plot_2d_points: int) -> None:
        Properties.__set_option('n_plot_2d_points', n_plot_2d_points)

    @staticmethod
    def set_n_plot_3d_points(n_plot_3d_points: int) -> None:
        Properties.__set_option('n_plot_3d_points', n_plot_3d_points)

    @staticmethod
    def set_n_time_grid_points(n_time_grid_points: int) -> None:
        Properties.__set_option('n_time_grid_points', n_time_grid_points)

    @staticmethod
    def set_abs_tol(value: float) -> None:
        Properties.__set_option('abs_tol', value)

    @staticmethod
    def set_rel_tol(value: float) -> None:
        Properties.__set_option('rel_tol', value)

    @staticmethod
    def get_prop_dict() -> Dict[str, Any]:
        return Properties.__conf_repo_mgr.copy()

    @staticmethod
    def __get_option(opt_name: str):
        # confRepMgr = elltool.conf.Properties.getConfRepoMgr();
        # opt = confRepMgr.getParam(optName);

        return Properties.__conf_repo_mgr[opt_name]

    @staticmethod
    def __set_option(opt_name: str, opt_val) -> None:
        # confRepMgr = elltool.conf.Properties.getConfRepoMgr();
        # confRepMgr.setParam(optName,optVal);
        
        Properties.__conf_repo_mgr[opt_name] = opt_val


# noinspection PyProtectedMember
Properties._implicit_init()
