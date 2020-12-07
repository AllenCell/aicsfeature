import pdb
import numpy as np
import pandas as pd
from scipy import stats
from skimage.morphology import skeletonize_3d as skskeletonize
from skimage.filters import gaussian as skgaussian
from skimage.measure import label as sklabel
from skimage.segmentation import find_boundaries as skfind_boundaries
from scipy.ndimage.morphology import distance_transform_edt as skdistance_transform_edt

from . import tools

_running_on_flag_ = None
_aicsfeature_debug_ = False

def get_normalized_image(img, scaling_params):

    img_norm = img.copy()
    mean = img_norm.mean()
    stdv = img_norm.std()
    strech_min = np.max([mean-scaling_params[0]*stdv, img_norm.min()])
    strech_max = np.min([mean+scaling_params[1]*stdv, img_norm.max()])
    img_norm[img_norm > strech_max] = strech_max
    img_norm[img_norm < strech_min] = strech_min
    img_norm = (img_norm-strech_min + 1e-8)/(strech_max-strech_min + 1e-8)
    img_norm = img_norm.clip(0,img_norm.max())

    return img_norm

def get_simple_binary_image():

    # Returns a simple 10x10x10 binary image

    img = np.zeros((10, 10, 10), dtype=np.uint8)
    img[4:7,4:7,4:7] = 1
    
    return img

def handle_connected_components_as_requested(input_image, mode):

    output_image = input_image.copy()

    output_image = sklabel(output_image)

    ncc = output_image.max()

    if ncc > 0:

        max_label = 1

        counts = np.bincount(output_image.reshape(-1))
        lcc = 1 + np.argmax(counts[1:])

        if mode == "pcc":

            max_label = output_image.max()

        elif mode == "lcc":

            output_image[output_image!=lcc] = 0
            output_image[output_image==lcc] = 1

        elif mode == "icc":

            output_image[output_image>0] = 1

    else:

        max_label = 0

    return ncc, max_label, output_image

#
# Apply a rotation of theta about the specified axis
#

def RotationMatrix(axis, theta):
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

#
# Calculates the PCA of input points (x,y,z) and
# align the axis from_axis (0, 1 or 2) to the
# cartesian axis to_axis (0, 1 or 2)
#

# def AlignPoints(x, y, z, from_axis, to_axis, align3d=False):
   
#     from sklearn.decomposition import PCA

#     df = pd.DataFrame({'x':x,'y':y,'z':z})

#     cartesian_axes = np.array([[1,0,0],[0,1,0],[0,0,1]])

#     eigenvecs = PCA(n_components=3).fit(df.values).components_

#     if align3d:

#         theta = np.arccos(np.clip(np.dot(eigenvecs[from_axis], cartesian_axes[to_axis]), -1.0, 1.0))

#         pivot = np.cross(eigenvecs[from_axis], cartesian_axes[to_axis])
        
#         rot_mx = RotationMatrix(pivot, theta)
        
#     else:

#         theta_proj = -np.arctan2(eigenvecs[0][1],eigenvecs[0][0])

#         rot_mx = [[np.cos(theta_proj),np.sin(-theta_proj),0],[np.sin(theta_proj),np.cos(theta_proj),0],[0,0,1]]
        
#     xyz_rot = np.dot(rot_mx, df.values.T).T

#     return xyz_rot[:,0], xyz_rot[:,1], xyz_rot[:,2]

def get_features(input_img, input_mask, info, running_on, input_mask2=None, input_mask3=None, meta_dict={}, debug=False):

    global _aicsfeature_debug_
    global _running_on_flag_
    _aicsfeature_debug_ = debug
    _running_on_flag_ = running_on

    #
    # Deal with connected components
    #

    number_cc, max_cc_label, input_mask = handle_connected_components_as_requested(input_image=input_mask, mode=info["mode"])

    #
    # Loop over every component
    #

    result = []

    if number_cc > 0:

        for cc in range(max_cc_label):

            features = {}

            input_mask_cc = input_mask.copy()
            input_mask_cc = (input_mask_cc==(cc+1)).astype(np.uint8)

            if "shape" in info:

                shape_features = get_shape_features(seg=input_mask_cc)

                features = {**features, **shape_features}

            if "roundness" in info:

                roundness_features = get_roundness_features(seg=input_mask_cc)

                features = {**features, **roundness_features}

            if "position" in info:

                position_features = get_position_features(seg=input_mask_cc, params=info["position"])

                features = {**features, **position_features}

            if "intensity" in info:
                
                input_masked = (input_img*input_mask_cc).astype(input_img.dtype)
                intensity_features = get_intensity_features(img=input_masked)

                features = {**features, **intensity_features}

            if "skeleton" in info:

                skeleton_features, _ = get_skeleton_features(seg=input_mask_cc)

                features = {**features, **skeleton_features}

            if "texture" in info:
                
                input_masked = (input_img*input_mask_cc).astype(input_img.dtype)
                texture_features, _ = get_texture_features(img=input_masked, params=info["texture"])

                features = {**features, **texture_features}

            if "io_intensity" in info:
                
                input_masked = (input_img*input_mask_cc).astype(input_img.dtype)
                io_intensity_features, _ = get_io_intensity_features(img=input_masked, params=info["io_intensity"])

                features = {**features, **io_intensity_features}

            if "bright_spots" in info:
                
                input_masked = (input_img*input_mask_cc).astype(input_img.dtype)
                bright_spots_features, _ = get_bright_spots_features(img=input_masked, params=info["bright_spots"])

                features = {**features, **bright_spots_features}

            if "profile" in info:

                if info["mode"] != "icc":
                    raise ValueError("Profile features are only available in icc mode")
                
                profile_features = get_profile_features(img=input_img, seg=input_mask_cc, params=info["profile"])
                
                features = {**features, **profile_features}

            if "dispersion" in info:

                if info["mode"] != "icc":
                    raise ValueError("Dispersion features are only available in icc mode")

                dispersion_features, _ = get_dispersion_features(seg1=input_mask, seg2=input_mask2, excluded_volume=input_mask3, params=info["dispersion"])

                features = {**features, **dispersion_features}

            if "shcoeffs" in info:

                if info["mode"] != "lcc":
                    raise ValueError("SH coeffs features are only available in lcc mode")

                shcoeffs_features, _ = get_shcoeffs_features(seg=input_mask, params=info["shcoeffs"])

                features = {**features, **shcoeffs_features}

            if "symmetry" in info:

                symmetry_features = get_symmetry_features(seg=input_mask, params=info["symmetry"])

                features = {**features, **symmetry_features}

            if "shexpansion" in info:

                if info["mode"] != "lcc":
                    raise ValueError("SH expansion features are only available in lcc mode")

                shexpansion_features, _ = get_shexpansion_features(seg=input_mask, params=info["shexpansion"])

                features = {**features, **shexpansion_features}

            if "neighborhood" in info:

                meta_features = get_neighborhood_features(meta=meta_dict)

                features = {**features, **meta_features}

            #
            #
            #

            result.append(features)

        result = pd.DataFrame(result, index=range(len(result)))

        if info["mode"] in ["lcc","pcc"]:

            result = result.add_suffix("_"+info["mode"])        

        if info["mode"] in ["pcc"]:

            result_std = result.std().add_suffix("_std")
            result_min = result.max().add_suffix("_max")
            result_max = result.min().add_suffix("_min")
            result_sum = result.sum().add_suffix("_sum")
            result_avg = result.mean().add_suffix("_avg")

            result = {**result_avg, **result_std, **result_min, **result_max, **result_sum}

            result = pd.DataFrame(result, index=[0])

        result = result.to_dict("records")[0]

        #
        # Connectivity
        #

        connectivity_features = {"connectivity_number_cc": number_cc}

        result = {**result, **connectivity_features}

    else:

        # If the input image is empty (number_cc = 0)

        result = {"connectivity_number_cc": number_cc}

    #
    # Final data frame
    #

    result = pd.DataFrame([result])

    result = result.add_prefix(info["prefix"]+"_")

    return result

#
# General
#

def get_shape_features(seg):

    """
        :param seg: 3D binary image containing a single connected
        component. Background has value 0 and object of interest
        has value > 0.

        :return: df - dictionary of features
    """

    features = {}

    if len(seg.shape) != 3:
        raise ValueError("Incorrect dimensions: {}".format(seg.shape))

    # Calculates the volume as the total number of non-zero voxels.

    features["shape_volume"] = np.count_nonzero(seg)

    # Calculates the axes features from the covariance matrix of the
    # voxels coordinates. Results are returned in descending order of
    # eigenvalue.

    z_pxl, y_pxl, x_pxl = np.where(seg > 0)

    number_of_voxels = len(z_pxl)

    # 2d area

    features["shape_2dmax_area"] = np.count_nonzero(seg.max(axis=0))

    features["shape_2dmid_area"] = np.count_nonzero(seg[int(z_pxl.mean())])

    axs = []
    all_eigen = []
    axs_length = []

    if number_of_voxels > 2:

        xyz_pxl_table = np.concatenate(
            [x_pxl.reshape(-1, 1), y_pxl.reshape(-1, 1), z_pxl.reshape(-1, 1)], axis=1
        )

        eigenvals, eigenvecs = np.linalg.eig(np.cov(xyz_pxl_table.transpose()))

        idx = eigenvals.argsort()
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]

        for i in range(3):
            vec = eigenvecs[:, -1 - i]
            ptp = np.inner(vec, xyz_pxl_table)
            axs_length.append(np.ptp(ptp))
            axs.append(vec)

        meridional_eccentricity = (
            np.nan
            if np.abs(eigenvals[2]) < 1e-12
            else eigenvals[1] / eigenvals[2]
        )
        equator_eccentricity = (
            np.nan
            if np.abs(eigenvals[1]) < 1e-12
            else eigenvals[0] / eigenvals[1]
        )

        eccentricity_31 = (
            np.nan
            if np.abs(eigenvals[2]) < 1e-12
            else eigenvals[0] / eigenvals[2]
        )
        eccentricity_21 = (
            np.nan
            if np.abs(eigenvals[2]) < 1e-12
            else eigenvals[1] / eigenvals[2]
        )
        eccentricity_32 = (
            np.nan
            if np.abs(eigenvals[1]) < 1e-12
            else eigenvals[0] / eigenvals[1]
        )

        features["shape_1st_axis_x"] = axs[0][0]
        features["shape_1st_axis_y"] = axs[0][1]
        features["shape_1st_axis_z"] = axs[0][2]
        features["shape_2nd_axis_x"] = axs[1][0]
        features["shape_2nd_axis_y"] = axs[1][1]
        features["shape_2nd_axis_z"] = axs[1][2]
        features["shape_3rd_axis_x"] = axs[2][0]
        features["shape_3rd_axis_y"] = axs[2][1]
        features["shape_3rd_axis_z"] = axs[2][2]
        features["shape_1st_axis_length"] = axs_length[0]
        features["shape_2nd_axis_length"] = axs_length[1]
        features["shape_3rd_axis_length"] = axs_length[2]
        features["shape_1st_eigenvalue"] = eigenvals[0]
        features["shape_2nd_eigenvalue"] = eigenvals[1]
        features["shape_3rd_eigenvalue"] = eigenvals[2]
        features["shape_3to1_eccentricity"] = eccentricity_31
        features["shape_2to1_eccentricity"] = eccentricity_21
        features["shape_3to2_eccentricity"] = eccentricity_32
        features["shape_meridional_eccentricity"] = meridional_eccentricity
        features["shape_equator_eccentricity"] = equator_eccentricity

    else:

        features["shape_1st_axis_x"] = np.nan
        features["shape_1st_axis_y"] = np.nan
        features["shape_1st_axis_z"] = np.nan
        features["shape_2nd_axis_x"] = np.nan
        features["shape_2nd_axis_y"] = np.nan
        features["shape_2nd_axis_z"] = np.nan
        features["shape_3rd_axis_x"] = np.nan
        features["shape_3rd_axis_y"] = np.nan
        features["shape_3rd_axis_z"] = np.nan
        features["shape_1st_axis_length"] = np.nan
        features["shape_2nd_axis_length"] = np.nan
        features["shape_3rd_axis_length"] = np.nan
        features["shape_1st_eigenvalue"] = np.nan
        features["shape_2nd_eigenvalue"] = np.nan
        features["shape_3rd_eigenvalue"] = np.nan
        features["shape_3to1_eccentricity"] = np.nan
        features["shape_2to1_eccentricity"] = np.nan
        features["shape_3to2_eccentricity"] = np.nan
        features["shape_meridional_eccentricity"] = np.nan
        features["shape_equator_eccentricity"] = np.nan

    return features

def get_roundness_features(seg):

    """
        :param seg: 3D binary image containing a single connected
        component. Background has value 0 and object of interest
        has value > 0.

        :return: df - dictionary of features
    """

    features = {}

    if len(seg.shape) != 3:
        raise ValueError("Incorrect dimensions: {}".format(seg.shape))

    from skimage.morphology import extrema, binary_erosion

    # Image with boundary voxels

    seg_raw = seg.copy()
    seg_raw[seg_raw>0] = 1

    # Forces a 1 pixel-wide offset to avoid problems with binary
    # erosion algorithm

    seg_raw[:,:,[0,-1]] = 0
    seg_raw[:,[0,-1],:] = 0
    seg_raw[[0,-1],:,:] = 0

    volume = seg_raw.sum()

    seg_surface = np.logical_xor(seg_raw, binary_erosion(seg_raw)).astype(np.uint8)

    # Loop through the boundary voxels to calculate the number of
    # boundary faces. Using 6-neighborhod.

    pxl_z, pxl_y, pxl_x = np.nonzero(seg_surface)

    dx = np.array([ 0, -1,  0,  1,  0,  0])
    dy = np.array([ 0,  0,  1,  0, -1,  0])
    dz = np.array([-1,  0,  0,  0,  0,  1])

    surface_area = 0
    for (k, j, i) in zip(pxl_z, pxl_y, pxl_x):
        surface_area += 6 - seg_raw[k+dz,j+dy,i+dx].sum()

    features["roundness_surface_area"] = surface_area

    if surface_area == 0:
        features["roundness_roughness"] = np.nan
        features["roundness_roughness_xy"] = np.nan
        features["roundness_roughness_xz"] = np.nan
        features["roundness_roughness_yz"] = np.nan
        return features

    # Calculates the sphericity that represents how closely the shape of the
    # object of interest approaches that of a mathematically perfect sphere.
    # Surface area of a discrete sphere is well approximated by
    #   S = exp(A)*V**B, where
    # S is the surface area, V is the volume (number of voxels) and the
    # constants A and B are 1.9851531 and 0.6664500, respectively.

    if features["roundness_surface_area"] == 0:

        features["roundness_sphericity"] = np.nan

    else:

        surface_area_sphere = np.exp(1.9851531)*(volume**0.6664500)

        features["roundness_sphericity"] = surface_area_sphere / features["roundness_surface_area"]

    # Roughness of a surface measures how ??

    # Centroid:

    zcm = np.int(pxl_z.mean())
    ycm = np.int(pxl_y.mean())
    xcm = np.int(pxl_x.mean())

    distance = []
    for (z, y, x) in zip(pxl_z, pxl_y, pxl_x):
        distance.append(np.sqrt((z-zcm)**2+(y-ycm)**2+(x-xcm)**2))
    distance = np.array(distance)

    distance_xy = distance[(pxl_z>(zcm-1)) & (pxl_z<(zcm+1))]
    distance_xz = distance[(pxl_y>(ycm-1)) & (pxl_y<(ycm+1))]
    distance_yz = distance[(pxl_x>(xcm-1)) & (pxl_x<(xcm+1))]

    features["roundness_roughness"] = distance.std() if len(distance) > 0 else np.nan
    features["roundness_roughness_xy"] = distance_xy.std() if len(distance_xy) > 0 else np.nan
    features["roundness_roughness_xz"] = distance_xz.std() if len(distance_xz) > 0 else np.nan
    features["roundness_roughness_yz"] = distance_yz.std() if len(distance_yz) > 0 else np.nan

    return features

def get_position_features(seg, params):

    """
        :param seg: 3D binary image containing a single connected
        component. Background has value 0 and object of interest
        has value > 0.

        :return: df - dictionary of features
    """

    #
    # Parameters
    #


    offset = [0,0,0] if "offset" not in params else params["offset"]

    #
    # Main
    #

    features = {}

    if len(seg.shape) != 3:
        raise ValueError("Incorrect dimensions: {}".format(seg.shape))

    z_pxl, y_pxl, x_pxl = np.nonzero(seg)

    z_pxl = z_pxl + offset[2]
    y_pxl = y_pxl + offset[1]
    x_pxl = x_pxl + offset[0]

    number_of_voxels = len(z_pxl)

    if number_of_voxels > 0:

        features["position_lowest_z"] = np.min(z_pxl)
        features["position_highest_z"] = np.max(z_pxl)
        features["position_x_centroid"] = np.mean(x_pxl)
        features["position_y_centroid"] = np.mean(y_pxl)
        features["position_z_centroid"] = np.mean(z_pxl)
        features["position_width"] = np.max(x_pxl) - np.min(x_pxl)
        features["position_height"] = np.max(y_pxl) - np.min(y_pxl)
        features["position_depth"] = np.max(z_pxl) - np.min(z_pxl)

    else:

        features["position_lowest_z"] = np.nan
        features["position_highest_z"] = np.nan
        features["position_x_centroid"] = np.nan
        features["position_y_centroid"] = np.nan
        features["position_z_centroid"] = np.nan
        features["position_width"] = np.nan
        features["position_height"] = np.nan
        features["position_depth"] = np.nan

    return features

def get_intensity_features(img):

    """
        :param seg: 3D 16-bit image (usually given by a multiplication
        of a gray scale image and its segmented version). The images
        contains a single connected component. Background has value 0
        and object of interest has value > 0.

        :return: df - dictionary of features
    """

    if len(img.shape) != 3:
        raise ValueError("Incorrect dimensions: {}".format(img.shape))

    features = {}

    # Pixel intensity moments and basic related statistics

    pxl_valids = np.nonzero(img)

    number_of_voxels = len(pxl_valids[0])

    if number_of_voxels > 0:

        features["intensity_mean"] = np.mean(img[pxl_valids])

        features["intensity_median"] = np.median(img[pxl_valids])

        features["intensity_sum"] = np.sum(img[pxl_valids])

        features["intensity_max"] = np.max(img[pxl_valids])

        features["intensity_std"] = np.std(img[pxl_valids])

        # Intensity entropy #requires some optimization

        prob = np.bincount(img[pxl_valids], minlength=65535)
        features["intensity_entropy"] = stats.entropy(prob / number_of_voxels)

    else:

        features["intensity_mean"] = np.nan
        features["intensity_median"] = np.nan
        features["intensity_sum"] = np.nan
        features["intensity_mode"] = np.nan
        features["intensity_max"] = np.nan
        features["intensity_std"] = np.nan
        features["intensity_entropy"] = np.nan

    return features

def get_skeleton_features(seg):

    """
        :param seg: 3D 16-bit image (usually given by a multiplication
        of a gray scale image and its segmented version). The images
        contains a single connected component. Background has value 0
        and object of interest has value > 0.

        :return: df - dictionary of features
    """

    features = {}

    if len(seg.shape) != 3:
        raise ValueError("Incorrect dimensions: {}".format(seg.shape))

    # 3D skeletonization from scikit image

    skel = skskeletonize(seg.astype(np.uint8))

    skel[skel > 0] = 1

    skel = np.pad(skel, 1, "constant")

    skel_degree = np.copy(skel)

    # Creating an image where the value of each pixel represents
    # its number of neighbors after skeletonization.

    z_pxl, y_pxl, x_pxl = np.where(skel > 0)

    nv = len(z_pxl)

    for x, y, z in zip(x_pxl, y_pxl, z_pxl):
        neigh = skel[z - 1 : z + 2, y - 1 : y + 2, x - 1 : x + 2]
        skel_degree[z, y, x] = neigh.sum()

    nt = skel.sum()
    n0 = np.sum(skel_degree == (0 + 1)) # n0 must be zero if pcc or lcc mode
    n1 = np.sum(skel_degree == (1 + 1))
    n2 = np.sum(skel_degree == (2 + 1))
    n3 = np.sum(skel_degree == (3 + 1))
    n4 = np.sum(skel_degree >= (4 + 1))

    # Average degree from <k> = Σ k x Pk

    if n2 != nt:
        average_degree = 0
        deg, Ndeg = np.unique(skel_degree.reshape(-1), return_counts=True)
        for k, n in zip(deg,Ndeg):
            if k != 2:
                average_degree = average_degree + k * (n / (nt-n2))
    else:
        average_degree = 1

    features["skeleton_voxels_number"] = nt
    features["skeleton_nodes_number"] = nt - n2
    features["skeleton_degree_mean"] = average_degree
    features["skeleton_edges_number"] = np.int(0.5 * (nt-n2) * average_degree)

    # Reminder: in the case of components that represent closed loops,
    # may only contain nodes with degree two. For sake of simplicity we
    # treat these single looped components as containing a single node
    # with degree one.

    features["skeleton_deg0_prop"] = (
        np.nan if nv == 0 else 0.0 if n2 == nt else n0 / (1.0 * nt - n2)
    )
    features["skeleton_deg1_prop"] = (
        np.nan if nv == 0 else 1.0 if n2 == nt else n1 / (1.0 * nt - n2)
    )
    features["skeleton_deg3_prop"] = (
        np.nan if nv == 0 else 0.0 if n2 == nt else n3 / (1.0 * nt - n2)
    )
    features["skeleton_deg4p_prop"] = (
        np.nan if nv == 0 else 0.0 if n2 == nt else n4 / (1.0 * nt - n2)
    )

    return features, (skel_degree)

def get_texture_features(img, params):

    import lmfit
    from matplotlib import cm, pyplot
    from skimage import feature as skfeature

    def GetCroppedCell(input_img, input_msk=None):
        
        if input_msk is None:
            [y,x] = np.nonzero(input_img)
        else:
            [y,x] = np.nonzero(input_msk)
        
        print(f"Effective Area: {x.size}")
            
        bins = np.percentile(input_img[y,x], [2,98])

        print(f"Bins for percentile normalization: {bins}")
        
        crop = input_img[y.min():y.max(),x.min():x.max()]
        
        crop[crop>0] = np.clip(crop[crop>0],*bins)
        
        return crop.astype(np.uint16)

    def QuantizeImage(input_img, nlevels=8):
        
        vmax = input_img.max()
        
        bins = [0] + np.percentile(input_img[input_img>0], np.linspace(0,100,nlevels+1))[:-1].tolist() + [1+vmax]
        
        input_dig = np.digitize(input_img, bins) - 1
        
        print(f"Frequency of each level: {np.bincount(input_dig.flatten())}")
        
        return input_dig

    def ExpDecay(x, a):
        return np.exp(-a*x**2)

    # Parameters

    nlevels = 8 if "nlevels" not in params else params["nlevels"]
    dmax = 32 if "dmax" not in params else params["dmax"]
    nangles = 16 if "nangles" not in params else params["nangles"]
    plot = False if "plot" not in params else params["plot"]
    save_fig = None if "save_fig" not in params else params["save_fig"]
    expdecay = 0.5

    # Main

    dists = np.linspace(0,dmax,dmax+1)
    angles = np.linspace(0,np.pi,nangles)

    crop = GetCroppedCell(img)
    crop = QuantizeImage(crop, nlevels=nlevels)

    glcm = skfeature.texture.greycomatrix(image=crop, distances=dists, angles=angles, levels=nlevels+1)
    corr = skfeature.texture.greycoprops(glcm[1:,1:], prop='correlation').T
    # corr = np.abs(corr)

    eps = 1e-1 # Regularizer for the mean to avoid too low values
    std_corr = corr.std(axis=0)
    avg_corr = eps + (1-eps)*corr.mean(axis=0)
    cvr_corr = std_corr / avg_corr
   
    if plot:

        fontsize = 16

        colormap = cm.get_cmap('cool',len(angles))
        
        print(f"Max coefficient of variation: {cvr_corr.max():1.3f}")

        fig, ax = pyplot.subplots(2,2,figsize=(12,8))

        ax[0,0].imshow(crop, cmap='gray')
        ax[0,0].axis('off')

        for i, curve in enumerate(corr):
            
            name = f'{180*angles[i]/np.pi:1.0f}°'
            
            ax[0,1].plot(dists,
                         curve,
                         color = colormap(i/(len(angles)-1)),
                         label = name,
                         linewidth = 1)
            
        ax[0,1].set_xlabel('Distance (pixels)', fontsize=fontsize)
        ax[0,1].set_ylabel('Correlation', fontsize=fontsize)
        ax[0,1].legend(bbox_to_anchor=(1.01, 1.05))
              
        ax[1,1].plot(dists,avg_corr,color='red',label='mean (reg)')
        ax[1,1].plot(dists,cvr_corr,color='blue',label='cv')
        ax[1,1].set_title(f'Maximum coefficient of variation: {cvr_corr.max():1.3f}')
        ax[1,1].set_xlabel('Distance (pixels)', fontsize=fontsize)
        ax[1,1].set_ylim(0,1)
        ax[1,1].legend()

        ax3 = ax[1,1].twinx()
        ax3.plot(dists,std_corr,color='black')
        ax3.set_ylim(0,std_corr.max())
        ax3.set_ylabel('StDev', fontsize=fontsize)

        pyplot.tight_layout()
        if save_fig is None:
            pyplot.show()
        else:
            fig.savefig(save_fig)
            pyplot.close(fig)
    
    features = {
        'texture_avg_corr': avg_corr,
        'texture_std_corr': std_corr
    }

    return features, crop

def get_texture_features_old(img, params):

    from mahotas.features import haralick

    """
        :param seg: 3D 16-bit image (usually given by a multiplication
        of a gray scale image and its segmented version). The images
        contains a single connected component. Background has value 0
        and object of interest has value > 0.

        :return: df - dictionary of features
    """

    distances = [1] if "distances" not in params else params["distances"]

    scaling_params = [0.5, 18] if "scaling_params" not in params else params["scaling_params"]

    # Main code

    features = {}

    if len(img.shape) != 3:
        raise ValueError("Incorrect dimensions: {}".format(img.shape))

    pxl_valids = np.nonzero(img)

    number_of_voxels = len(pxl_valids[0])

    img_norm = get_normalized_image(img, scaling_params)

    # Haralick requires integer type data

    img_norm = (255*(img_norm/img_norm.max())).astype(np.uint8)

    # Haralick texture features as decribed in [1]. See [2] for original paper by Haralick et. al
    # [1] - https://mahotas.readthedocs.io/en/latest/api.html?highlight=mahotas.features.haralick
    # [2] - Haralick et. al. Textural features for image classification. IEEE Transactions on systems, man, and cybernetics, (6), 610-621.
    # Notice that a minimal number of pixels (512) is required for computing these features.

    number_of_voxels_required = 512

    features = {
            "texture_haralick_ang2nd_moment": [],
            "texture_haralick_contrast": [],
            "texture_haralick_corr": [],
            "texture_haralick_variance": [],
            "texture_haralick_inv_diff_moment": [],
            "texture_haralick_sum_mean": [],
            "texture_haralick_sum_var": [],
            "texture_haralick_sum_entropy": [],
            "texture_haralick_entropy": [],
            "texture_haralick_diff_var": [],
            "texture_haralick_diff_entropy": [],
            "texture_haralick_info_corr1": [],
            "texture_haralick_info_corr2": [],
            "texture_distance_used": []
    }

    if number_of_voxels >= number_of_voxels_required:

        for d in distances:

            features["texture_distance_used"].append(d)

            ftextural = haralick(img_norm, ignore_zeros=True, return_mean=True, distance=d)

            for f_id, f_name in enumerate([
                    "texture_haralick_ang2nd_moment",
                    "texture_haralick_contrast",
                    "texture_haralick_corr",
                    "texture_haralick_variance",
                    "texture_haralick_inv_diff_moment",
                    "texture_haralick_sum_mean",
                    "texture_haralick_sum_var",
                    "texture_haralick_sum_entropy",
                    "texture_haralick_entropy",
                    "texture_haralick_diff_var",
                    "texture_haralick_diff_entropy",
                    "texture_haralick_info_corr1",
                    "texture_haralick_info_corr2"
                ]):

                features[f_name].append(ftextural[f_id])

    return features, (img_norm)

def get_dispersion_features(seg1, seg2, excluded_volume=None, params={}):

    """
        Calculates the dispersion of objects in image1
        wrt objects in image2. Low values of dispersion
        indicate objects in image1 is more clustered
        than objects in image2.

        :param seg1: 3D binary image. Background has
        value 0 and object of interest has value > 0.
        :param seg2: 3D binary image. Background has
        value 0 and object of interest has value > 0.
        seg2 should corresponds to the large portion
        of the image. Usually corresponds to cytoplasm.

        :param normalize: if the coordinates should be
        normalized such that the largest distance to the
        origin is unitary.
        :param number_of_samples: number of sampled points
        drawn from image1 and image2 at each repetition.
        :param repetitions: number of realizations of the
        calculation. Results are averaged over each
        repetition.
        :return: df - pandas dictionary of features
    """

    #
    # Parameters
    #

    normalize = 1 if "normalize" not in params else params["normalize"]

    number_of_samples = 50 if "number_of_samples" not in params else params["number_of_samples"]

    repetitions = 400 if "repetitions" not in params else params["repetitions"]

    #
    # Main
    #

    features = []

    if len(seg1.shape) != 3:
        raise ValueError("Incorrect dimensions: {}".format(seg1.shape))

    if len(seg2.shape) != 3:
        raise ValueError("Incorrect dimensions: {}".format(seg2.shape))

    if (seg2>0).sum() < (seg1>0).sum():
        raise ValueError("Objects in seg2 have larger volume than objects in seg1. Aborting...")

    reference = 1

    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment

    # Finding nonzero voxels

    coords = []
    coords.append(np.nonzero(seg1))

    if type(excluded_volume) is np.ndarray:
        coords.append(np.nonzero( (seg2>0).astype(np.uint8) - (excluded_volume>0).astype(np.uint8) ))
    else:
        coords.append(np.nonzero(seg2))

    for i in range(2):
        coords[i] = np.asarray(coords[i]).T
        if _aicsfeature_debug_:
            print("#Points in set {0} = {1}.".format(i,coords[i].shape))

    dist_matrix_str = None
    dist_matrix_ctl = None

    if coords[0].shape[0] > number_of_samples and coords[1].shape[0] > number_of_samples:

        centroid = []
        diameter = []
        for i in range(2):
            centroid.append(coords[i].mean(axis=0))
            diameter.append(np.sqrt(((coords[i]-centroid[i])**2).sum(axis=1)).max())
            if _aicsfeature_debug_:
                print("#Set {0}. Centroid = {1}, Diameter = {2}.".format(i,centroid[i],diameter[i]))

        # Shifting to the origin of reference set

        for i in range(2):
            coords[i] = coords[i] - centroid[reference]

        # Normalization by the diameter

        if normalize:
            for i in range(2):
                coords[i] /= diameter[reference]
                
        # Sampling points

        for realization in range(repetitions):
        
            samples = []
            for i in [reference,1-reference,reference]:
                idx = np.random.choice(np.arange(coords[i].shape[0]), size=number_of_samples, replace=False)
                samples.append(coords[i][idx,:])

            # Matrix of distance between sampled points

            dist_matrix_str = cdist(samples[0], samples[1])
            dist_matrix_ctl = cdist(samples[0], samples[2])

            # Solving the bipartite graph matching

            match_str_row, match_str_col = linear_sum_assignment(dist_matrix_str)
            match_ctl_row, match_ctl_col = linear_sum_assignment(dist_matrix_ctl)

            # Solving the closest neighbor matching

            match_str_local_col = np.argmin(dist_matrix_str,axis=1)
            match_str_local_row = np.argmin(dist_matrix_str,axis=0)
            match_ctl_local_col = np.argmin(dist_matrix_ctl,axis=1)
            match_ctl_local_row = np.argmin(dist_matrix_ctl,axis=0)

            # Calculating cost dispersion values

            cost_str_mean = dist_matrix_str[match_str_row, match_str_col].mean()
            cost_ctl_mean = dist_matrix_ctl[match_ctl_row, match_ctl_col].mean()
            cost_str_local_mean = np.mean([dist_matrix_str[np.arange(len(match_str_local_col)), match_str_local_col],dist_matrix_str[match_str_local_row, np.arange(len(match_str_local_col))]])
            cost_ctl_local_mean = np.mean([dist_matrix_ctl[np.arange(len(match_ctl_local_col)), match_ctl_local_col],dist_matrix_ctl[match_ctl_local_row, np.arange(len(match_ctl_local_col))]])

            cost_str_std = dist_matrix_str[match_str_row, match_str_col].std()
            cost_ctl_std = dist_matrix_ctl[match_ctl_row, match_ctl_col].std()
            cost_str_local_std = np.std([dist_matrix_str[np.arange(len(match_str_local_col)), match_str_local_col],dist_matrix_str[match_str_local_row, np.arange(len(match_str_local_col))]])
            cost_ctl_local_std = np.std([dist_matrix_ctl[np.arange(len(match_ctl_local_col)), match_ctl_local_col],dist_matrix_ctl[match_ctl_local_row, np.arange(len(match_ctl_local_col))]])

            features.append({"dispersion_cost_data": cost_str_mean,
                             "dispersion_cost_local_data": cost_str_local_mean,
                             "dispersion_data": cost_str_std,
                             "dispersion_local_data": cost_str_local_std,
                             "dispersion_cost_control": cost_ctl_mean,
                             "dispersion_cost_local_control": cost_ctl_local_mean,
                             "dispersion_control": cost_ctl_std,
                             "dispersion_local_control": cost_ctl_local_std})

    else:

        features.append({"dispersion_cost_data": np.nan,
                         "dispersion_cost_local_data": np.nan,
                         "dispersion_data": np.nan,
                         "dispersion_local_data": np.nan,
                         "dispersion_cost_control": np.nan,
                         "dispersion_cost_local_control": np.nan,
                         "dispersion_control": np.nan,
                         "dispersion_local_control": np.nan})

    features = pd.DataFrame(features)
    features = features.mean().to_dict()

    return features, (dist_matrix_str,dist_matrix_ctl)

#
# Nucleus project
#

def get_io_intensity_features(img, params):

    """
        :img: 3D 16-bit image (usually given by a multiplication
        of a gray scale image and its segmented version). The images
        contains a single connected component. Background has value 0
        and object of interest has value > 0.

        :params:
            {
                "radii": list N of integers values,
                "ratio_rings": list of 2 integers values
            }
        Applies erosion operation N times
        to divide the image into N regions. We report
        values for regions specified in ratio_rings.

                 1.  2.  3.    4.         5
        surface |-|----|----|----|----------------| center

        :return: df - dictionary of features
    """

    radii = [1,4,4,4] if "radii" not in params else params["radii"]

    ratio_rings = [2,4] if "ratio_rings" not in params else params["ratio_rings"]

    # Main code

    features = {}

    if len(img.shape) != 3:
            raise ValueError("Incorrect dimensions: {}".format(img.shape))

    from skimage.morphology import binary_erosion

    img_bin = img.copy()
    img_bin[img_bin>0] = 1

    # Sequence of erosion to create inner, mid and outer images

    img_rings = img_bin.copy()
    for r in radii:
        for i in range(r):
            img_bin = binary_erosion(img_bin)
        img_rings += img_bin

    img_outer = img_rings.copy()
    img_outer = (img_outer==ratio_rings[0]).astype(np.uint8)

    img_inner = img_rings.copy()
    img_inner = (img_inner==ratio_rings[1]).astype(np.uint8)

    # we do not compute anything if the inner region has size zero

    if img_inner.sum() > 0:

        # Center slice

        z, _, _ = np.nonzero(img)

        center_slice = np.int(z.mean())

        for img_region, region_name in zip([img_inner,img_outer],["inner","outer"]):

            # Non zero voxel of each region

            z, y, x = np.nonzero(img_region)

            number_of_voxels = len(z)

            fea_name = "io_intensity_"+region_name+"_volume"
            features[fea_name] = number_of_voxels

            average_intensity = (img*img_region)[z,y,x].mean()

            fea_name = "io_intensity_"+region_name+"_mean"
            features[fea_name] = average_intensity

            # Non zero voxels in the center slice

            x_slice = x[ (z>(center_slice-1)) & (z<(center_slice+1)) ]
            y_slice = y[ (z>(center_slice-1)) & (z<(center_slice+1)) ]
            z_slice = z[ (z>(center_slice-1)) & (z<(center_slice+1)) ]

            average_intensity_slice = (img*img_region)[z_slice,y_slice,x_slice].mean()

            fea_name = "io_intensity_"+region_name+"_slice_mean"
            features[fea_name] = average_intensity_slice

    else:

        for img_region, region_name in zip([img_inner,img_outer],["inner","outer"]):

            fea_name = "io_intensity_"+region_name+"_volume"
            features[fea_name] = np.nan

            fea_name = "io_intensity_"+region_name+"_mean"
            features[fea_name] = np.nan

            fea_name = "io_intensity_"+region_name+"_slice_mean"
            features[fea_name] = np.nan

    return features, (img_rings)

def get_bright_spots_features(img, params):

    """
        Uses extrema.h_maxima from skimage.morphology
        to identify bright spots in the input image.
        After detection, a region around each maxima
        is cropped to create an average spot from which
        measurements are taken.

        :param seg: 3D 16-bit image (usually given by a multiplication
        of a gray scale image and its segmented version). The images
        contains a single connected component. Background has value 0
        and object of interest has value > 0.

        :return: df - dictionary of features
    """

    #
    # Parameters
    #

    sigma = 1 if "sigma" not in params else params["sigma"]

    radii = 11 if "radii" not in params else params["radii"]

    spot_threshold = 0.85 if "spot_threshold" not in params else params["spot_threshold"]

    scaling_params = [0.5, 18] if "scaling_params" not in params else params["scaling_params"]

    #
    # Main
    #

    features = {}

    if len(img.shape) != 3:
        raise ValueError("Incorrect dimensions: {}".format(img.shape))

    from skimage.morphology import extrema, binary_dilation

    img_norm = get_normalized_image(img=img, scaling_params=scaling_params)

    img_norm = gaussian(image=img_norm, sigma=sigma)


    # Find maxima

    img_max = extrema.h_maxima(img_norm, h=0.1)
    z_pxl, y_pxl, x_pxl = np.nonzero(img_max)

    # For each maxima we crop a region and append their max
    # projection

    spots = []
    coords = []
    for n_id in range(len(x_pxl)):

        x = x_pxl[n_id]
        y = y_pxl[n_id]
        z = z_pxl[n_id]
        
        # check whether the neighborhod falls inside the image

        if (np.min([radii,x,y,img.shape[2]-x-1,img.shape[1]-y-1]) == radii) and (z > 2) and (z < img.shape[0]-3):
            
            # region round the bright spot

            img_spot = img[(z-2):(z+3),(y-radii):(y+radii+1),(x-radii):(x+radii+1)]
            
            # append in a list

            spots.append(img_spot.max(axis=0))

            # save coordinates

            coords.append((x,y,z))

        else:

            img_max[z,y,x] = 0

    # list to array

    spots = np.array(spots)
    spots.reshape(-1,2*radii+1,2*radii+1)

    coords = np.array(coords)

    # for each spot

    intensity = []

    cross_sec_area = []

    number_of_spots = spots.shape[0]

    for spot in range(number_of_spots):

        # first we check whether this spot has any neighbor spot within
        # a sphere with readius r

        xo = coords[spot,0]
        yo = coords[spot,1]
        zo = coords[spot,2]

        dist = np.sqrt((coords[:,0]-xo)**2+(coords[:,1]-yo)**2+(coords[:,2]-zo)**2)

        # only considers spots with no near neighbors

        if not (np.sum(dist<radii) > 1):

            img_spot = spots[spot].copy()

            # normalize to max 1 and apply threshold

            max_intensity = img_spot.max()
            img_spot = img_spot / max_intensity
            img_spot_bin = (img_spot>spot_threshold).astype(np.uint8)

            # find the largest connected component that includes the central pixel

            img_spot_mask = sklabel(img_spot_bin, connectivity=1)
            img_spot_mask = (img_spot_mask==img_spot_mask[radii,radii]).astype(np.uint8)

            cross_sec_area.append(img_spot_mask.sum())

            # Avg intensity in the spot

            img_spot = img_spot * max_intensity * img_spot_mask

            intensity.append(img_spot[img_spot>0].mean())

    if len(cross_sec_area) > 0:

        features["bright_spots_number"] = number_of_spots

        features["bright_spots_number_used"] = len(cross_sec_area)

        features["bright_spots_intensity_mean"] = np.mean(intensity)

        features["bright_spots_xy_cross_sec_area_mean"] = np.mean(cross_sec_area)

        features["bright_spots_xy_cross_sec_area_std"] = np.std(cross_sec_area)

    else:

        features["bright_spots_number"] = 0

        features["bright_spots_number_used"] = 0

        features["bright_spots_intensity_mean"] = np.nan

        features["bright_spots_xy_cross_sec_area_mean"] = np.nan

        features["bright_spots_xy_cross_sec_area_std"] = np.nan

    return features, (coords,cross_sec_area)

def get_profile_features(img, seg, params):

    """
        Calculate image properties as a function of z (assumed to be
        the first axis).
        :param img: 3D 16-bit image
        :param seg: 3D 16-bit labeled image with one or more connected
        components. Background has value 0 and objects of interest has
        value > 0.

        :return: df - dictionary of features
    """

    #
    # Parameters
    #

    # ...

    #
    # Main
    #

    features = {}

    if len(img.shape) != 3:
        raise ValueError("Incorrect dimensions: {}".format(img.shape))

    #
    # Intensity
    #        

    features["profile_intensity"] = list(img.mean(axis=-1).mean(axis=-1))

    #
    # Connected components
    #

    from scipy import ndimage

    seg_cc = ndimage.label(seg)[0]
    centroids = ndimage.measurements.center_of_mass(seg, seg_cc, np.arange(1,1+seg_cc.max()).tolist())

    features["profile_centroid_z"] = list(np.bincount([int(z[0]) for z in centroids], minlength=seg.shape[0]))

    return features

def get_shcoeffs_features(seg, params):

    import pyshtools
    from skimage.transform import resize as skresize
    from scipy.interpolate import griddata as scigriddata

    """
        Need some doc here.

        :return: df - dictionary of features
    """

    #
    # Parameters
    #

    min_size = 128 # pixels

    align = 'align2d' if "align" not in params else params["align"]

    sigma = None if "sigma" not in params else params["sigma"]

    lmax = 8 if "lmax" not in params else params["lmax"]

    #
    # Main
    #

    features = {}

    if len(seg.shape) != 3:
        raise ValueError("Incorrect dimensions: {}".format(seg.shape))

    #
    # Features names
    #

    l_labels = np.repeat(np.arange(lmax+1).reshape(-1,1),lmax+1,axis=1)
    m_labels = np.repeat(np.arange(lmax+1).reshape(-1,1),lmax+1,axis=1).T
    l_labels = l_labels.reshape(-1)
    m_labels = m_labels.reshape(-1)

    lm_labels  = ['shcoeffs_L{:d}M{:d}C'.format(l,m) for (l,m) in zip(l_labels,m_labels)]
    lm_labels += ['shcoeffs_L{:d}M{:d}S'.format(l,m) for (l,m) in zip(l_labels,m_labels)]

    lm_labels += ['shcoeffs_energy{:d}'.format(l) for l in range(1,lmax+1)]

    ft_labels = lm_labels + ['shcoeffs_chi2']

    #
    # Main
    #

    if _aicsfeature_debug_:
        print("Running on: {0}".format(_running_on_flag_))
        pdb.set_trace()

    seg[seg>0] = 1

    # Size threshold

    if seg.sum() < min_size:
        for f in ft_labels:
            features[f] = np.nan
        return features, None

    mesh, (img,cm) = tools.GetPolyDataFromNumpy(volume=seg, lcc=False, sigma=sigma, center=True, size_threshold=min_size)

    transformation = dict()
    for i, ax in enumerate(['xc','yc','zc']):
        transformation[ax] = cm[i]

    if mesh is None:
        return features, None

    x, y, z = [], [], []
    for i in range(mesh.GetNumberOfPoints()):
        r = mesh.GetPoints().GetPoint(i)
        x.append(r[0])
        y.append(r[1])
        z.append(r[2])
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    if align == 'align2d':
        x, y, z, align_params = tools.AlignPoints(x,y,z)
    elif align == 'align3d':
        x, y, z, align_params = tools.AlignPoints(x, y, z, from_axis=0, to_axis=0, align3d=True)
        x, y, z, align_params = tools.AlignPoints(x, y, z, from_axis=1, to_axis=1, align3d=True)

    transformation.update(align_params)

    for i in range(mesh.GetNumberOfPoints()):
        mesh.GetPoints().SetPoint(i,x[i],y[i],z[i])
    mesh.Modified()

    r = np.sqrt(x**2+y**2+z**2)
    lat = np.arccos(np.divide(z, r, out=np.zeros_like(r), where=r!=0))
    lon = np.pi + np.arctan2(y,x)

    imethod = 'nearest'
    points = np.concatenate([np.array(lon).reshape(-1,1),np.array(lat).reshape(-1,1)], axis=1)
    grid_lon, grid_lat = np.meshgrid(
        np.linspace(start=0, stop=2*np.pi, num=256, endpoint=True),
        np.linspace(start=0, stop=  np.pi, num=128, endpoint=True)
    )
    grid = scigriddata(points, r, (grid_lon, grid_lat), method=imethod)

    # Fit grid data with SH

    coeffs = pyshtools.expand.SHExpandDH(grid, sampling=2, lmax_calc=lmax)

    if coeffs[0,0,0] < 1e-5:
        for f in ft_labels:
            features[f] = np.nan
        return features, None

    # Reconstruct grid

    grid_rec = pyshtools.expand.MakeGridDH(coeffs, sampling=2)

    # Compute rec error

    grid_down = skresize(grid,output_shape=grid_rec.shape, preserve_range=True)

    chi2 = np.power(grid_down-grid_rec, 2).mean()

    energy = np.square(np.abs(coeffs)/coeffs[0,0,0])
    energy = energy.sum(axis=0).sum(axis=1)

    features = np.concatenate((np.concatenate((coeffs[0].ravel(),coeffs[1].ravel())),energy[1:],[chi2],[*transformation.values()]))

    # return dict

    features = pd.DataFrame(features.reshape(1,-1))

    features.columns = ft_labels + ['shcoeffs_transform_'+k for k in transformation.keys()]
    
    return features.to_dict("records")[0], (img, mesh, cm, grid_down, grid_rec, coeffs, chi2)

def get_symmetry_features(seg, params):

    def quit_feature():
        for ax in ['z','y','x']:
            features['symmetry_'+ax] = np.nan
        return features

    def get_symmetry(img_aligned, axis):

        img = np.rollaxis(img_aligned, axis)
        profile = img.sum(axis=-1).sum(axis=-1).astype(np.float64)
        profile_f = profile[1:-1].copy()
        sym = []
        for (i,j) in zip([0,1,2],[-2,-1,None]):
            profile_b = profile[i:j].copy()
            profile_b = profile_b[::-1]
            sym.append(1 - np.abs(profile_f-profile_b).sum()/img_bin.sum())    
        return np.max(sym)

    """
        Need some doc here.

        :return: df - dictionary of features
    """

    #
    # Parameters
    #

    min_size = 4

    sigma = None if "smooth" not in params else params["smooth"]

    align = 'align2d' if "align" not in params else params["align"]

    #
    # Main
    #

    features = {}

    if len(seg.shape) != 3:
        raise ValueError("Incorrect dimensions: {}".format(seg.shape))

    img_bin = seg.copy()

    if sigma:
        img_bin = img_bin.astype(np.float32)
        img_bin = skgaussian(img_bin,sigma=(sigma,sigma,sigma))
        img_bin[img_bin<1.0/np.exp(1.0)] = 0
        img_bin[img_bin>0] = 1
        img_bin = img_bin.astype(np.uint8)

    z, y, x = np.nonzero(img_bin)

    if len(x) < min_size:
        return quit_feature()

    xo = x.mean()
    yo = y.mean()
    zo = z.mean()

    x = x - xo
    y = y - yo
    z = z - zo


    if align == 'align2d':

        x, y, z, _ = tools.AlignPoints(x,y,z)
        if np.any(np.isnan(x)):
            return quit_feature()

    elif align == 'align3d':
        x, y, z, _ = tools.AlignPoints(x, y, z, from_axis=0, to_axis=0, align3d=True)
        if np.any(np.isnan(x)):
            return quit_feature()
        x, y, z, _ = tools.AlignPoints(x, y, z, from_axis=1, to_axis=1, align3d=True)
        if np.any(np.isnan(x)):
            return quit_feature()

    img_rot = np.zeros_like(img_bin)

    extx = 2 * np.abs(x).max()
    padx = int(extx-img_rot.shape[2]+1.5) if extx >= img_rot.shape[2] else 0
    exty = 2 * np.abs(y).max()
    pady = int(exty-img_rot.shape[1]+1.5) if exty >= img_rot.shape[1] else 0
    extz = 2 * np.abs(z).max()
    padz = int(extz-img_rot.shape[0]+1.5) if extz >= img_rot.shape[0] else 0

    if np.array([padx,pady,padz]).any():
        img_rot = np.pad(img_rot, ((padz,padz),(pady,pady),(padx,padx)), 'constant')

    xo = int(0.5*img_rot.shape[2])
    yo = int(0.5*img_rot.shape[1])
    zo = int(0.5*img_rot.shape[0])

    img_rot[[int(v+zo) for v in z],[int(v+yo) for v in y],[int(v+xo) for v in x]] = 1

    for axis, ax in zip([0,1,2],['z','y','x']):

        features['symmetry_'+ax] = get_symmetry(img_aligned=img_rot, axis=axis)

    return features

def get_shexpansion_features(seg, params):
    
    import vtk
    from vtk.util import numpy_support
    import igraph
    import pyshtools
    from scipy import sparse
    from scipy.interpolate import griddata
    from scipy.sparse.linalg import spsolve

    """
        This function performs a spherical harmonic expansion of an
        object in a binary image. First the surface of this object is
        represented by a triangular mesh. Next we use spherical
        parametrization to expand X(theta,phi), Y(theta,phi) and
        Z(theta,phi) up to the order Lmax of spherical harmonics.

        :return: df - dictionary of features containing the coefficients
        of the expansion
    """

    #
    # Auxiliar function
    #

    def solve_heat_equation_on_graph(graph, init_ids, init_values):

        global _running_on_flag_

        n = graph.vcount()
        
        source, target = zip(*graph.get_edgelist())

        I = source + target
        J = target + source
        V = -np.ones(len(I))

        A = sparse.coo_matrix((V,(I,J)),shape=(n,n), dtype=np.float64)
        K = graph.degree()
        A.setdiag(K)

        A = A.toarray()

        # Ideally would delete rows and cols without converting the matrix to dense
        A = np.delete(A, init_ids, axis=0) # remove init_ids with boundary condition
        A = np.delete(A, init_ids, axis=1) # remove init_ids with boundary condition

        '''
            # Alternative solution for the two lines above that does not
            require the matrix to be converted to dense. It still need to
            be tested.

            import numpy as np
            from scipy.sparse import csr_matrix

            def delete_from_csr(mat, row_indices=[], col_indices=[]):
                """
                Remove the rows (denoted by ``row_indices``) and columns (denoted by ``col_indices``) from the CSR sparse matrix ``mat``.
                WARNING: Indices of altered axes are reset in the returned matrix
                """
                if not isinstance(mat, csr_matrix):
                    raise ValueError("works only for CSR format -- use .tocsr() first")

                rows = []
                cols = []
                if row_indices:
                    rows = list(row_indices)
                if col_indices:
                    cols = list(col_indices)

                if len(rows) > 0 and len(cols) > 0:
                    row_mask = np.ones(mat.shape[0], dtype=bool)
                    row_mask[rows] = False
                    col_mask = np.ones(mat.shape[1], dtype=bool)
                    col_mask[cols] = False
                    return mat[row_mask][:,col_mask]
                elif len(rows) > 0:
                    mask = np.ones(mat.shape[0], dtype=bool)
                    mask[rows] = False
                    return mat[mask]
                elif len(cols) > 0:
                    mask = np.ones(mat.shape[1], dtype=bool)
                    mask[cols] = False
                    return mat[:,mask]
                else:
                    return mat
        '''

        A = sparse.csr_matrix(A)

        b = np.zeros(n, dtype=np.float64)
        c = np.zeros(n, dtype=np.int)
        for vid, value in zip(init_ids,init_values):
            vidn = graph.neighborhood(vertices=vid, order=1, mode=igraph.ALL) # includes source node
            c[vidn] += 1
            b[vidn] += value
                
        b = np.delete(b, init_ids)
        c = np.delete(c, init_ids)
        b[c>1] /= c[c>1] # normalizing nodes with multiple neighbors with initial condition > 0

        solution = np.zeros(n)
        solution[np.delete(np.arange(n),init_ids)] = spsolve(A, b)
        solution[init_ids] = init_values
            
        # histogram-based latitude equalization
        h, solution_reg = np.histogram(solution, 180, density=True)
        cdf = h.cumsum()
        cdf = np.max(init_values) * (cdf-cdf[0]) / (cdf[-1]-cdf[0])
        solution = np.interp(solution, solution_reg[:-1], cdf)
        
        return solution

    def GetiGraphFromPolyData(mesh):
        
        # Create the graph and add vertices
        graph = igraph.Graph(directed=False)
        nv = mesh.GetNumberOfPoints()
        nc = mesh.GetNumberOfCells()
        graph.add_vertices(nv)
        
        coords = []
        for i in range(nv):
            ri = mesh.GetPoints().GetPoint(i)
            coords.append(ri)
        coords = np.array(coords)
        
        # Store coords and vtkids as nodes properties
        graph.vs['x'] = coords[:,0]
        graph.vs['y'] = coords[:,1]
        graph.vs['z'] = coords[:,2]
        graph.vs['vtkid'] = np.arange(nv).tolist()
        
        # Add edges
        edges = []
        length = []
        for cid in range(nc):
            cell = mesh.GetCell(cid)
            for eid in range(cell.GetNumberOfEdges()):
                edge = cell.GetEdge(eid)
                i = edge.GetPointId(0)
                j = edge.GetPointId(1)
                ri = np.array(mesh.GetPoints().GetPoint(i))
                rj = np.array(mesh.GetPoints().GetPoint(j))
                length.append(np.sqrt(np.square(ri-rj).sum()))
                edges.append((i,j))
        graph.add_edges(edges)
        graph.es['length'] = length
        graph = graph.simplify(multiple=True, loops=True, combine_edges='first')
        return graph

    def smooth_scalar_field(graph, scalar, fixset, eps=1e-3):
        
        # calculate the mse of each vertex scalar compared to
        # the average of its neighbors.
        
        tofix = []
        err_min = 1e5
        for vid in range(graph.vcount()):
            if vid not in fixset:
                neighs = graph.neighborhood(vertices=vid, order=1)
                err = np.square(scalar[neighs[0]]-np.mean(scalar[neighs[1:]]))
                if err > eps:
                    tofix.append(vid)
                err_min = err if err < err_min else err_min

        neighs = graph.neighborhood(vertices=tofix, order=1)
        for i, vid in enumerate(tofix):
            scalar[vid] = np.mean(scalar[neighs[i][1:]])
        
        return scalar, len(tofix)

    def simplify_mesh(polydata, target_reduction=0.9, n_smooth_iter=256):
        dec = vtk.vtkDecimatePro()
        dec.SetInputData(polydata)
        dec.PreserveTopologyOn()
        dec.SetTargetReduction(target_reduction)
        dec.Update()

        smooth = vtk.vtkSmoothPolyDataFilter()
        smooth.SetInputData(dec.GetOutput())
        smooth.SetNumberOfIterations(256)
        smooth.Update()
       
        return smooth.GetOutput()

    def get_polydata_from_array(input_image, lcc=True):

        input_image = input_image.astype(np.float32)

        imagedata = vtk.vtkImageData()
        imagedata.SetDimensions(input_image.shape)

        input_image = input_image.transpose(2,1,0)
        input_image_output = input_image.copy(0)
        input_image = input_image.flatten()
        arr = numpy_support.numpy_to_vtk(input_image, array_type=vtk.VTK_FLOAT)
        arr.SetName('Scalar')
        imagedata.GetPointData().SetScalars(arr)

        cf = vtk.vtkContourFilter()
        cf.SetInputData(imagedata)
        cf.SetValue(0, 1.0/np.exp(1.0))
        cf.Update()

        if lcc:

            conn = vtk.vtkConnectivityFilter()
            conn.SetInputData(cf.GetOutput())
            conn.SetExtractionModeToLargestRegion()
            conn.Update()

            clean = vtk.vtkCleanPolyData()
            clean.SetInputData(conn.GetOutput())
            clean.Update()

            polydata = clean.GetOutput()

        else:

            polydata = cf.GetOutput()

        return polydata

    def map_scalar_to_polydata(polydata, ids, scalars, name):

        n = polydata.GetNumberOfPoints()
        arr = vtk.vtkFloatArray()
        arr.SetNumberOfComponents(1)
        arr.SetNumberOfTuples(n)
        arr.FillComponent(0,0)
        arr.SetName(name)
        for vid, scalar in zip(ids,scalars):
            arr.SetTuple1(vid,scalar)
        polydata.GetPointData().AddArray(arr)
        polydata.Modified()
               
        return polydata

    #
    # Parameters
    #

    min_size = 16 # pixels

    flip_y = 1 if "flip_y" not in params else params["flip_y"]

    sigma = None if "sigma" not in params else params["sigma"]

    lmax = 8 if "lmax" not in params else params["lmax"]

    map_scalars = 0 if "map_scalars" not in params else params["map_scalars"]

    return_full = 0 if "return_full" not in params else params["return_full"]

    reduction = 0.9

    #
    # Features names
    #

    l_labels = np.repeat(np.arange(lmax+1).reshape(-1,1),lmax+1,axis=1)
    m_labels = np.repeat(np.arange(lmax+1).reshape(-1,1),lmax+1,axis=1).T
    l_labels = l_labels.reshape(-1)
    m_labels = m_labels.reshape(-1)

    ft_labels  = ['shexpansion_L{:d}M{:d}C'.format(l,m) for (l,m) in zip(l_labels,m_labels)]
    ft_labels += ['shexpansion_L{:d}M{:d}S'.format(l,m) for (l,m) in zip(l_labels,m_labels)]
    if return_full:
        ft_labels += ['shexpansion_XL{:d}M{:d}C'.format(l,m) for (l,m) in zip(l_labels,m_labels)]
        ft_labels += ['shexpansion_XL{:d}M{:d}S'.format(l,m) for (l,m) in zip(l_labels,m_labels)]
        ft_labels += ['shexpansion_YL{:d}M{:d}C'.format(l,m) for (l,m) in zip(l_labels,m_labels)]
        ft_labels += ['shexpansion_YL{:d}M{:d}S'.format(l,m) for (l,m) in zip(l_labels,m_labels)]
        ft_labels += ['shexpansion_ZL{:d}M{:d}C'.format(l,m) for (l,m) in zip(l_labels,m_labels)]
        ft_labels += ['shexpansion_ZL{:d}M{:d}S'.format(l,m) for (l,m) in zip(l_labels,m_labels)]
    ft_labels += ['shexpansion_energy{:d}'.format(l) for l in range(1,lmax+1)]

    #
    # Main
    #

    features = {}
    for f in ft_labels:
        features[f] = np.nan

    if len(seg.shape) != 3:
        raise ValueError("Incorrect dimensions: {}".format(seg.shape))

    img = seg.copy()

    # Make sure the image borders are empty
    img[:,:, 0] = 0
    img[:,:,-1] = 0
    img[:, 0,:] = 0
    img[:,-1,:] = 0
    img[:,:, 0] = 0
    img[:,:,-1] = 0

    img[img>0] = 1

    # Size threshold

    if img.sum() < min_size:

        for f in ft_labels:

            features[f] = np.nan

        return features, None

    # Pre-processing

    if flip_y:
        img = img[:,::-1,:]
    
    if sigma:
        img = img.astype(np.float32)
        img = skgaussian(img,sigma=(sigma,sigma,sigma))


    mesh = get_polydata_from_array(input_image=img)

    # ---------- Rotation

    # a = 180*np.random.rand()
    # u = np.random.randn(3,1)
    # u /= np.linalg.norm(u, axis=0)

    # xm, ym, zm = 0., 0., 0.
    # for i in range(mesh.GetNumberOfPoints()):
    #     r = mesh.GetPoints().GetPoint(i)
    #     xm += r[0]
    #     ym += r[1]
    #     zm += r[2]
    # xm /= mesh.GetNumberOfPoints()
    # ym /= mesh.GetNumberOfPoints()
    # zm /= mesh.GetNumberOfPoints()

    # transform = vtk.vtkTransform()
    # transform.PostMultiply()
    # transform.Translate(-xm,-ym,-zm)
    # transform.RotateWXYZ(a,u[0],u[1],u[2])
    # transform.Translate(+xm,+ym,+zm)
    # transform.Update()

    # transformer = vtk.vtkTransformPolyDataFilter()
    # transformer.SetInputData(mesh)
    # transformer.SetTransform(transform)
    # transformer.Update()

    # mesh = transformer.GetOutput()

    # ----------

    if _aicsfeature_debug_:
        from skimage import io as skio
        skio.imsave('debug_extractor_shexpansion_input_img.tif', img.astype(np.float32))
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(mesh)
        writer.SetFileName('debug_extractor_shexpansion_mesh_raw.vtk')
        writer.Write()

    mesh = simplify_mesh(polydata=mesh, target_reduction=reduction)

    if _aicsfeature_debug_:
        writer.SetInputData(mesh)
        writer.SetFileName('debug_extractor_shexpansion_mesh.vtk')
        writer.Write()

    # get graph representation
    g = GetiGraphFromPolyData(mesh)

    n = g.vcount()
    f = mesh.GetNumberOfCells()
    e = g.ecount()

    if n-e+f != 2:
        print("Running on {0}. SH expansion only works on genus-zero meshes. 2-2G={1}, N={2}, E={3}, F={4}".format(_running_on_flag_,n-e+f,n,e,f))
        if _aicsfeature_debug_:
            pdb.set_trace()
        return features, (img,mesh,None,None)
    
    # calculate diameter of the graph
    diameter = g.get_diameter(directed=False, weights='length')

    if map_scalars:
        mesh = map_scalar_to_polydata(polydata=mesh,
                                      ids=diameter,
                                      scalars=[1]*len(diameter),
                                      name='diameter')

    # solve heat equation for latitude
    init = [np.pi,0]
    ids = [diameter[0],diameter[-1]]

    if _aicsfeature_debug_:
        pdb.set_trace()

    lat = solve_heat_equation_on_graph(g,ids,init)

    if map_scalars:
        mesh = map_scalar_to_polydata(polydata=mesh,
                                      ids=range(n),
                                      scalars=lat,
                                      name='latitude')

    gbkp = g.copy()

    # remove north and south pole edges and their neighbors edges
    edges = []
    for i in [diameter[0],diameter[-1]]:
        neighs_i = g.neighborhood(vertices=i, order=1)[1:]
        edges += [(i,j) for j in neighs_i]
        for j in neighs_i:
            neighs_j = g.neighborhood(vertices=j, order=1)[1:]
            edges += [(j,k) for k in neighs_j]
    g.delete_edges(edges)

    if _aicsfeature_debug_:
        pdb.set_trace()

    # find cc on the left and right side of the diameter
    neighs = g.neighborhood(vertices=diameter, order=1)
    neighs = [vid for sublist in neighs for vid in sublist]
    neighs = [vid for vid in neighs if vid not in diameter]
    subg = g.subgraph(neighs)

    cc = subg.clusters()

    if len(cc) == 0:
        print("Running on {0}. Subgraph subg must have at least 2 connected components. Number ofcomponents found = {1}".format(_running_on_flag_,len(cc)))
        return features, None

    # heated nodes belong to the largest cc in case there are more than 2 ccs
    heat_sources_cc = 1 if len(cc) == 2 else np.argsort(cc.sizes())[-1]

    if _aicsfeature_debug_:
        pdb.set_trace()

    # remove edges between diameter and cc on the right side
    g = gbkp.copy()
    meridian = [vid for i,vid in enumerate(subg.vs()['vtkid']) if cc.membership[i] == heat_sources_cc]
    for source in diameter:
        neighs = g.neighborhood(vertices=source, order=1)
        neighs = [vid for vid in neighs if vid in meridian]
        edges = [(source,vid) for vid in neighs[1:]]
        g.delete_edges(edges)

    if map_scalars:
        mesh = map_scalar_to_polydata(polydata=mesh,
                                      ids=subg.vs()['vtkid'],
                                      scalars=cc.membership,
                                      name='meridian')
    if _aicsfeature_debug_:
        pdb.set_trace()

    # solve heat equation for longitude
    ids = diameter + meridian
    init = [0] * len(diameter) + [2*np.pi] * len(meridian)
    ids, init = list(zip(*sorted(zip(ids,init))))
    lon = solve_heat_equation_on_graph(g,list(ids),list(init))

    if map_scalars:
        mesh = map_scalar_to_polydata(polydata=mesh,
                                      ids=range(n),
                                      scalars=lon,
                                      name='longitude_raw')
    if _aicsfeature_debug_:
        pdb.set_trace()

    while True:
        lon, nerr = smooth_scalar_field(graph=g, scalar=lon, fixset=diameter+meridian, eps=1e-3)
        if not nerr: break

    if map_scalars:
        mesh = map_scalar_to_polydata(polydata=mesh,
                                      ids=range(n),
                                      scalars=lon,
                                      name='longitude')

    xyz = []
    for i in range(n):
        r = mesh.GetPoints().GetPoint(i)
        xyz.append([r[0],r[1],r[2]])

    if _aicsfeature_debug_:
        pdb.set_trace()

    imethod = 'nearest'
    xyz = np.array(xyz)
    points = np.concatenate([np.array(lon).reshape(-1,1),np.array(lat).reshape(-1,1)], axis=1)
    grid_lon, grid_lat = np.meshgrid(
        np.linspace(start=0, stop=2*np.pi, num=256, endpoint=True),
        np.linspace(start=0, stop=  np.pi, num=128, endpoint=True)
    )
    grid_x = griddata(points, xyz[:,0], (grid_lon, grid_lat), method=imethod)
    grid_y = griddata(points, xyz[:,1], (grid_lon, grid_lat), method=imethod)
    grid_z = griddata(points, xyz[:,2], (grid_lon, grid_lat), method=imethod)

    if _aicsfeature_debug_:
        pdb.set_trace()

    coeffs_x = pyshtools.expand.SHExpandDH(grid_x, sampling=2, lmax_calc=lmax)
    coeffs_y = pyshtools.expand.SHExpandDH(grid_y, sampling=2, lmax_calc=lmax)
    coeffs_z = pyshtools.expand.SHExpandDH(grid_z, sampling=2, lmax_calc=lmax)

    coeffs = np.sqrt(np.square(coeffs_x)+np.square(coeffs_y)+np.square(coeffs_z))

    energy = np.square(np.abs(coeffs/coeffs[0,0,0])).sum(axis=0).sum(axis=1)

    features = np.concatenate((coeffs[0].ravel(),coeffs[1].ravel()))
    if return_full:
        features = np.concatenate((features,coeffs_x[0].ravel(),coeffs_x[1].ravel()))
        features = np.concatenate((features,coeffs_y[0].ravel(),coeffs_y[1].ravel()))
        features = np.concatenate((features,coeffs_z[0].ravel(),coeffs_z[1].ravel()))
    features = np.concatenate((features,energy[1:]))
   
    # return dict

    features = pd.DataFrame(features.reshape(1,-1))

    features.columns = ft_labels

    return features.to_dict("records")[0], (img, mesh, (coeffs_x,coeffs_y,coeffs_z), (grid_x,grid_y,grid_z))

def get_neighborhood_features(meta):

    """
        Need some doc here.

        :return: df - dictionary of features
    """

    #
    # Main
    #

    features = {}

    list_of_neigh_features = [
        ('this_cell_nbr_dist_2d','dist2d'),
        ('this_cell_nbr_dist_3d','dist3d'),
        ('this_cell_nbr_overlap_area','contact')
    ]

    for var, name in list_of_neigh_features:

        var_dict = eval(meta[var])

        values = [v for (CellId,v) in var_dict]

        if len(values):

            features[ f'neighborhood_{name}_min'] =  np.min(values)
            features[ f'neighborhood_{name}_max'] =  np.max(values)
            features[ f'neighborhood_{name}_sum'] =  np.sum(values)
            features[f'neighborhood_{name}_mean'] = np.mean(values)
            features[f'neighborhood_{name}_median'] = np.median(values)

        else:

            features[ f'neighborhood_{name}_min'] = np.nan
            features[ f'neighborhood_{name}_max'] = np.nan
            features[ f'neighborhood_{name}_sum'] = np.nan
            features[f'neighborhood_{name}_mean'] = np.nan
            features[f'neighborhood_{name}_median'] = np.nan

    features['neighborhood_number_neighbors'] = len(values)

    return features
