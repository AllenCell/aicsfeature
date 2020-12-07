import aicsimageprocessing as proc
import numpy as np
import pandas as pd
import math


def get_features(im_nuc, im_cell, im_struct):

    """
    Gets features related to the structure localization in the cell and nuclear regions
    :param im_nuc: binary segmentation of nucleus
    :param im_cell: binary segmentation of cell
    :param im_struct: grayscale image of the structure of interest

    :return: df - pandas dictionary of features
    """
    im_nuc = im_nuc > 0
    im_cell = im_cell > 0

    assert np.all(im_nuc == (im_nuc * im_cell))

    fp = polarity_features(im_nuc, im_cell, im_struct)

    df = pd.DataFrame(fp, index=[0])

    df = df.add_prefix("dna_cell_struct_")

    return df


def polarity_features(im_nuc, im_cell, im_struct):

    """
        Gets features related to the structure localization in the cell and nuclear regions
        :param im_nuc: binary segmentation of nucleus
        :param im_cell: binary segmentation of cell
        :param im_struct: grayscale image of the structure of interest

        :return: features - dictionary of feature names and their values
    """

    features = {}

    im_struct = im_struct * im_cell
    im_cyto = im_cell * (im_nuc == 0)

    im_ratio, im_th, im_phi, _, _ = proc.img_to_coords(im_cell, im_nuc)

    ####
    # Radial affinity
    ####

    # get the 50th prctile of distance from the nuclear membrane outward (cytoplasmic region)
    ratio_prctile = np.percentile(im_ratio[im_cyto], 50)
    cyto_distal_inten = np.sum(im_struct[(im_ratio > ratio_prctile) * im_cyto])
    features["cyto_distal_ratio"] = cyto_distal_inten / np.sum(im_struct[im_cyto])

    # get the 50th prctile of distance from the nuclear membrane inward (nuclear region)
    ratio_prctile = np.percentile(im_ratio[im_nuc], 50)
    nuc_distal_inten = np.sum(im_struct[(im_ratio < ratio_prctile) * im_nuc])
    features["nuc_distal_ratio"] = nuc_distal_inten / np.sum(im_struct[im_nuc])

    # get the cytoplasmic affinity (nuc vs cyto region)
    cell_distal_inten = np.sum(im_struct[im_cyto])
    features["cell_distal_ratio"] = cell_distal_inten / sum(im_struct[im_cell])

    ####
    # Z-affinity
    ####

    # Because of images, we take two masks and average between them

    features["cyto_z_ratio"] = 0
    features["nuc_z_ratio"] = 0
    features["cell_z_ratio"] = 0

    z_masks = [im_phi > 0, im_phi >= 0]

    for z_mask in z_masks:
        cyto_z_inten = np.sum(im_struct[z_mask & im_cyto])
        features["cyto_z_ratio"] += (cyto_z_inten / np.sum(im_struct[im_cyto])) / len(
            z_masks
        )

        nuc_z_inten = np.sum(im_struct[z_mask & im_nuc])
        features["nuc_z_ratio"] += (nuc_z_inten / np.sum(im_struct[im_nuc])) / len(
            z_masks
        )

        cell_z_inten = np.sum(im_struct[z_mask & im_cell])
        features["cell_z_ratio"] += (cell_z_inten / np.sum(im_struct[im_cell])) / len(
            z_masks
        )

    ####
    # Maj axis split
    ####

    features["cyto_maj_ratio"] = 0
    features["nuc_maj_ratio"] = 0
    features["cell_maj_ratio"] = 0

    th_masks = [im_th >= 0, im_th > 0]

    for th_mask in th_masks:

        cyto_maj_inten = np.sum(im_struct[th_mask & im_cyto])
        features["cyto_maj_ratio"] += (
            cyto_maj_inten / np.sum(im_struct[im_cyto])
        ) / len(th_masks)

        nuc_maj_inten = np.sum(im_struct[th_mask & im_nuc])
        features["nuc_maj_ratio"] += (nuc_maj_inten / np.sum(im_struct[im_nuc])) / len(
            th_masks
        )

        cell_maj_inten = np.sum(im_struct[th_mask & im_cell])
        features["cell_maj_ratio"] += (
            cell_maj_inten / np.sum(im_struct[im_cell])
        ) / len(th_masks)

    ####
    # Min axis split
    ####

    im_min_mask = (im_th < (math.pi / 2)) & (im_th > -(math.pi / 2))

    cyto_min_inten = np.sum(im_struct[im_min_mask & im_cyto])
    features["cyto_min_ratio"] = cyto_min_inten / np.sum(im_struct[im_cyto])

    nuc_min_inten = np.sum(im_struct[im_min_mask & im_nuc])
    features["nuc_min_ratio"] = nuc_min_inten / np.sum(im_struct[im_nuc])

    cell_min_inten = np.sum(im_struct[im_min_mask & im_cell])
    features["cell_min_ratio"] = cell_min_inten / np.sum(im_struct[im_cell])

    # adjust all features to be from -1 to 1
    for f in features:
        f_tmp = (features[f] - 0.5) * 2

        # take abs value for left/right or front/back features
        if "min" in f or "maj" in f:
            f_tmp = np.abs(f_tmp)

        features[f] = f_tmp

    return features


def get_list_of_features():

    im_nuc = np.zeros((10, 10, 10))
    im_nuc[3:7, 3:7, 3:7] = 1

    im_cell = np.zeros((10, 10, 10))
    im_cell[1:-1, 1:-1, 1:-1] = 1

    df = get_features(im_nuc=im_nuc, im_cell=im_cell, im_struct=np.ones((10, 10, 10)))

    return df.columns.values


if __name__ == "__main__":

    """
        Testing features
    """

    print(get_list_of_features())
