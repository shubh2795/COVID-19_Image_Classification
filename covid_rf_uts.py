import unittest
from random_forests_and_decision_trees import *


class mnist_dt_rf_uts(unittest.TestCase):

    def test_ut01(self):
        fit_validate_dt()

    def test_ut02(self):
        ndts = 5
        fit_validate_dts(ndts)

    def test_ut03(self):
        fit_validate_rf(5)

    def test_ut04(self):
        low_nt, high_nt = 10, 100
        fit_validate_rfs(low_nt, high_nt)


if __name__ == '__main__':
    unittest.main()
    pass
