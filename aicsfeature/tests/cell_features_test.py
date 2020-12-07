import unittest
import numpy as np

from aicsfeature.extractor import cell


class cell_features_test(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.__init__(self)

        dims = 50
        hdim = 5

        im_cell = np.zeros([dims, dims, dims])
        im_cell[hdim:-hdim, hdim:-hdim, hdim:-hdim] = 1

        self.im_cell = im_cell.astype("uint16")

    def test_cell_features_plz(self):
        # plz work
        cell.get_features(self.im_cell)

        self.assertTrue(True)
