import unittest
import numpy as np

from aicsfeature.extractor import dna


class dna_features_test(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.__init__(self)

        dims = 50
        hdim = 5

        im_dna = np.zeros([dims, dims, dims])
        im_dna[hdim:-hdim, hdim:-hdim, hdim:-hdim] = 1

        self.im_dna = im_dna.astype("uint16")

    def test_cell_features_plz(self):
        # plz work
        dna.get_features(self.im_dna)

        self.assertTrue(True)
