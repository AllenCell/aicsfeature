import numpy as np
import pandas as pd
from .common import get_simple_binary_image, get_shape_features, get_position_features, get_intensity_features, get_texture_features, get_skeleton_features


def get_features(img, extra_features=[]):

    """
        Returns shape and position features averaged over
        all connected components of a given structure. In
        addition to mean values, this function returns the
        std, min and max.

        :param img: 3D binary image containing a structure
        with one or more connected components. Background
        has value 0 and structure components have pixel
        value > 0

        :extra_features: list of additional features to be
        computed.

        :return: df - pandas dictionary of features
    """

    from skimage.measure import label, regionprops

    # Detecting connected components

    bin = img.copy()
    bin[bin > 0] = 1
    regions = label(bin)
    props = regionprops(regions)
    ncomps = len(props)

    # Calculating features for each individual component

    Features = []

    for pid, prop in enumerate(props):

        comp = np.copy(regions[prop.bbox[0]:prop.bbox[3],
                               prop.bbox[1]:prop.bbox[4],
                               prop.bbox[2]:prop.bbox[5]])
        comp[comp!=(pid+1)] = 0

        fs = get_shape_features(seg=comp)
        fp = get_position_features(seg=comp)

        features = {**fs, **fp}

        if "intensity" in extra_features:
            fextra = get_intensity_features(img=comp)
            features = {**features, **fextra}

        if "skeleton" in extra_features:
            fextra = get_skeleton_features(seg=comp)
            features = {**features, **fextra}

        if "texture" in extra_features:
            fextra = get_texture_features(img=comp)
            features = {**features, **fextra}

        Features.append(features)

    # When no structure signal is found in the image.

    if ncomps == 0:

        fs = get_shape_features(seg=img)
        fp = get_position_features(seg=img)
        features = {**fs, **fp}
        if "intensity" in extra_features:
            fextra = get_intensity_features(img=img)
            features = {**features, **fextra}
        if "skeleton" in extra_features:
            fextra = get_skeleton_features(seg=img)
            features = {**features, **fextra}
        if "texture" in extra_features:
            fextra = get_texture_features(img=img)
            features = {**features, **fextra}
        Features.append(features)

    # Final one-row dataframe corresponding to basic stats over all components.

    Features = pd.DataFrame(Features)

    Favg = Features.mean().add_prefix("str_").add_suffix("_mean")
    Fstd = Features.std().add_prefix("str_").add_suffix("_std")
    Fmin = Features.max().add_prefix("str_").add_suffix("_min")
    Fmax = Features.min().add_prefix("str_").add_suffix("_max")
    Fsum = Features.sum().add_prefix("str_").add_suffix("_sum")

    Features = {**Favg, **Fstd, **Fmin, **Fmax, **Fsum}

    Features["str_components_number"] = ncomps

    df = pd.DataFrame([Features], index=[0])
    
    return df

def print_list_of_features():

    # Simple binary data to calculate the features

    img = get_simple_binary_image()

    df0 = get_features(img=img)

    print("\nbasic\n")

    print(df0.columns.values[:,None])

    for extra in ["skeleton", "texture", "intensity"]:

        dfe = get_features(img=img, extra_features=[extra])

        print("\n",extra,"\n")

        print(dfe.columns.difference(df0.columns).values[:,None])


if __name__ == "__main__":

    # Listing available features

    print_list_of_features()

