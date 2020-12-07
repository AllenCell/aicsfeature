import unittest
import numpy as np

from aicsfeature.extractor import cell_nuc
from scipy.ndimage.morphology import distance_transform_edt as bwdist
from skimage.transform import resize


class polarity_features_test(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.__init__(self)

        dims = 201
        h_dim = int((dims - 1) / 2)

        im_bounding_box = np.ones([dims, dims, dims])
        im_bounding_box[h_dim, h_dim, h_dim] = 0
        im_dist = bwdist(im_bounding_box)
        im_dist = resize(im_dist, [h_dim + 1, h_dim + 1, dims], order=0)

        self.im_cell = im_dist <= (h_dim - 2)
        self.im_nuc = im_dist <= np.percentile(im_dist[self.im_cell], 50)

    def test_polarity_features_uniform_cell(self):
        # tests features for uniform whole cell signal
        polarity_features = cell_nuc.polarity_features(
            self.im_nuc, self.im_cell, self.im_cell
        )

        for k in polarity_features:
            np.testing.assert_almost_equal(np.isnan(polarity_features[k]), 0, 2)

    def test_polarity_features_uniform_nuc(self):
        # tests features for uniform nuclear signal
        polarity_features = cell_nuc.polarity_features(
            self.im_nuc, self.im_cell, self.im_nuc
        )

        for k in polarity_features:
            if "cyto_" in k:
                self.assertTrue(np.isnan(polarity_features[k]))
            elif "nuc_" in k:
                np.testing.assert_almost_equal(polarity_features[k], 0, 2)

        self.assertEqual(polarity_features["cell_distal_ratio"], -1)
        self.assertEqual(polarity_features["cell_z_ratio"], 0)
        np.testing.assert_almost_equal(polarity_features["cell_maj_ratio"], 0, 2)
        np.testing.assert_almost_equal(polarity_features["cell_min_ratio"], 0, 2)

    def test_polarity_features_uniform_cyto(self):
        # tests features for uniform cytoplasmic signal
        im_cyto = self.im_cell.copy()
        im_cyto[self.im_nuc] = 0

        polarity_features = cell_nuc.polarity_features(
            self.im_nuc, self.im_cell, im_cyto
        )

        for k in polarity_features:
            if "cyto_" in k:
                self.assertAlmostEqual(np.isnan(polarity_features[k]), 0, places=2)
            if "nuc_" in k:
                self.assertTrue(np.isnan(polarity_features[k]))

        self.assertEqual(polarity_features["cell_distal_ratio"], 1)
        self.assertEqual(polarity_features["cell_z_ratio"], 0)
        np.testing.assert_almost_equal(polarity_features["cell_maj_ratio"], 0, 2)
        np.testing.assert_almost_equal(polarity_features["cell_min_ratio"], 0, 2)
