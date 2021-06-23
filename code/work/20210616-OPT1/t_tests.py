# RA, 2021-06-23

import unittest
import datetime
import time

from b_casestudent import get_default_params
from z_sources import get_problem_data


class TestSources(unittest.TestCase):
    def test_get_problem_data_deterministic(self):
        params = get_default_params()['data']
        params['max_trips'] = 10

        pdata1 = get_problem_data(**params)
        pdata2 = get_problem_data(**params)

        self.assertTrue(pdata1.trips.equals(pdata2.trips))
