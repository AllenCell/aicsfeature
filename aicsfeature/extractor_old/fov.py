import numpy as np
import pandas as pd
from .common import get_simple_binary_image, get_shape_features, get_position_features, get_texture_features


def get_features(img):

    """
        Returns position features for FOV images

        :param img: 3D binary image containing many
        connected components. Background has value 0 and
        structure components have pixel value > 0

        :return: df - pandas dictionary of features
    """

    fp = get_position_features(img=seg)

    df = pd.DataFrame(fp, index=[0])

    df = df.add_prefix("fov_")

    return df


def print_list_of_features():

    # Returns a list with features names

    img = get_simple_binary_image()

    df = get_features(seg=img)

    print(df.columns.values[:,None])

if __name__ == "__main__":

    # Testing features

    print_list_of_features()
