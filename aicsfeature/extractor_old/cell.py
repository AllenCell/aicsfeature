import pandas as pd
from .common import (
    get_simple_binary_image,
    get_shape_features,
    get_position_features,
    get_intensity_features,
    get_texture_features,
)


def get_features(img, seg=None):

    """
        Returns shape, position, texture and intensity features
        for images of Cell.

        :param img: 3D binary image containing a single cell
        object. Background has value 0 and structure components
        have pixel value > 0

        :return: df - pandas dictionary of features
    """
    if seg is None:
        seg = img > 0
    else:
        img[seg == 0] = 0

    fs = get_shape_features(seg=seg)

    fp = get_position_features(seg=seg)

    fi = get_intensity_features(img=img)

    ft = get_texture_features(img=img)

    df = pd.DataFrame({**fs, **fp, **fi, **ft}, index=[0])

    df = df.add_prefix("cell_")

    return df


def print_list_of_features():

    # Simple binary data to calculate the features

    img = get_simple_binary_image()

    df = get_features(img=img)

    print(df.columns.values[:, None])


if __name__ == "__main__":

    # Listing available features

    print_list_of_features()
