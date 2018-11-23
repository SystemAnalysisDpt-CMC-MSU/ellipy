import numpy as np
from products.ellipy.gras.sym.sym import is_dependent


class TestSym:
    def test_is_dependent(self):
        assert is_dependent(np.array([['cos(t)', 'sin(t)'], ['-sin(t)', 'cost(t)']], dtype='str'))
        assert not is_dependent(np.array([['cos(t)', 'sin(t)'], ['-sin(t)', 'cost(t)']], dtype='str'), is_discrete=True)
        assert is_dependent(np.array([['cos', 'k'], ['-sin', 'cosk']], dtype='str'), is_discrete=True)
        assert not is_dependent(np.array([['cos', 'k'], ['-sin', 'cosk']], dtype='str'))
