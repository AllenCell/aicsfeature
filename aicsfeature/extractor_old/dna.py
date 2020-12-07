import pandas as pd
from .common import (get_simple_binary_image,
                    get_shape_features,
                    get_position_features,
                    get_intensity_features,
                    get_texture_features,
                    get_io_intensity_features,
                    get_bright_spots_features,
                    get_roundness_features)


def get_features(img, extra_features=[], seg=None):

    """
        Returns shape, position and texture/intensity features
        for images of DNA.

        :param img: 3D image containing a single connected DNA
        objects. Background has value 0 and structure components
        have pixel value > 0

        :extra_features: list of additional features to be
        computed.

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

    features = {**fs, **fp, **fi, **ft}

    if "io_intensity" in extra_features:
        fextra = get_io_intensity_features(img=img, number_ops=4)
        features = {**features, **fextra}

    if "bright_spots" in extra_features:
        fextra = get_bright_spots_features(img=img)
        features = {**features, **fextra}

    if "roundness" in extra_features:
        fextra = get_roundness_features(seg=img)
        features = {**features, **fextra}

    df = pd.DataFrame(features, index=[0])

    df = df.add_prefix("dna_")

    return df


def print_list_of_features():

    # Simple binary data to calculate the features

    img = get_simple_binary_image()

    df0 = get_features(img=img)

    print(df0.columns.values[:, None])

    for extra in ["io_intensity", "bright_spots", "roundness"]:

        dfe = get_features(img=img, extra_features=[extra])

        print("\n", extra, "\n")

        print(dfe.columns.difference(df0.columns).values[:, None])


if __name__ == "__main__":

    # Listing available features

    print_list_of_features()
