import numpy as np

from scipy import stats
from scipy.ndimage.morphology import distance_transform_edt
from skimage.morphology import skeletonize_3d

from mahotas.features import haralick


def get_simple_binary_image():

    # Returns a simple 10x10x10 binary image

    img = np.zeros((10, 10, 10), dtype=np.uint8)
    img[4:7, 4:7, 4:7] = 1

    return img


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

    features["volume"] = np.count_nonzero(seg)

    # Calculates the axes features from the covariance matrix of the
    # voxels coordinates. Results are returned in descending order of
    # eigenvalue.

    z_pxl, y_pxl, x_pxl = np.where(seg > 0)

    number_of_voxels = len(z_pxl)

    axs = []
    axs_length = []

    if number_of_voxels:

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
            else np.sqrt(1 - np.square(eigenvals[0] / eigenvals[2]))
        )
        equator_eccentricity = (
            np.nan
            if np.abs(eigenvals[1]) < 1e-12
            else np.sqrt(1 - np.square(eigenvals[0] / eigenvals[1]))
        )

    if number_of_voxels == 0:

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
        features["shape_meridional_eccentricity"] = np.nan
        features["shape_equator_eccentricity"] = np.nan

    else:

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
        features["shape_meridional_eccentricity"] = meridional_eccentricity
        features["shape_equator_eccentricity"] = equator_eccentricity

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

    features["surface_area"] = surface_area

    # Calculates the sphericity that represents how closely the shape of the
    # object of interest approaches that of a mathematically perfect sphere.
    # Surface area of a discrete sphere is well approximated by
    #   S = exp(A)*V**B, where
    # S is the surface area, V is the volume (number of voxels) and the
    # constants A and B are 1.9851531 and 0.6664500, respectively.

    if features["surface_area"] == 0:

        features["roundness_sphericity"] = np.nan

    else:

        surface_area_sphere = np.exp(1.9851531)*(volume**0.6664500)

        features["roundness_sphericity"] = surface_area_sphere / features["surface_area"]

    # Roughness of a surface measures how ??

    # Centroid:

    zcm = pxl_z.mean()
    ycm = pxl_y.mean()
    xcm = pxl_x.mean()

    distance = []
    for (z, y, x) in zip(pxl_z, pxl_y, pxl_x):
        distance.append(np.sqrt((z-zcm)**2+(y-ycm)**2+(x-xcm)**2))
    distance = np.array(distance)

    features["roundness_roughness"] = distance.std()

    features["roundness_roughness_xy"] = distance[(pxl_z>(zcm-1)) & (pxl_z<(zcm+1))].std()

    features["roundness_roughness_xz"] = distance[(pxl_y>(ycm-1)) & (pxl_y<(ycm+1))].std()

    features["roundness_roughness_yz"] = distance[(pxl_x>(xcm-1)) & (pxl_x<(xcm+1))].std()

    return features

def get_position_features(seg):

    """
        :param seg: 3D binary image containing a single connected
        component. Background has value 0 and object of interest
        has value > 0.

        :return: df - dictionary of features
    """

    features = {}

    if len(seg.shape) != 3:
        raise ValueError("Incorrect dimensions: {}".format(seg.shape))

    z_pxl, y_pxl, x_pxl = np.nonzero(seg)

    number_of_voxels = len(z_pxl)

    if number_of_voxels > 0:

        features["position_lowest_z"] = np.min(z_pxl)
        features["position_highest_z"] = np.max(z_pxl)
        features["position_x_centroid"] = np.mean(x_pxl)
        features["position_y_centroid"] = np.mean(y_pxl)
        features["position_z_centroid"] = np.mean(z_pxl)

    else:

        features["position_lowest_z"] = np.nan
        features["position_highest_z"] = np.nan
        features["position_x_centroid"] = np.nan
        features["position_y_centroid"] = np.nan
        features["position_z_centroid"] = np.nan

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

        features["intensity_mode"] = stats.mode(img[pxl_valids])[0][0]

        features["intensity_max"] = np.max(img[pxl_valids])

        features["intensity_std"] = np.std(img[pxl_valids])

        # Intensity entropy

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


def get_texture_features(img, scaling_params=[0.5, 18]):

    """
        :param seg: 3D 16-bit image (usually given by a multiplication
        of a gray scale image and its segmented version). The images
        contains a single connected component. Background has value 0
        and object of interest has value > 0.

        :return: df - dictionary of features
    """

    features = {}

    if len(img.shape) != 3:
        raise ValueError("Incorrect dimensions: {}".format(img.shape))

    pxl_valids = np.nonzero(img)

    number_of_voxels = len(pxl_valids[0])

    # Contrast normalization for Haralick texture features. Using
    # Jianxu's code implemented in aics-segmentation

    scaling_params = [0.5, 18]

    img_norm = img.copy()
    mean = img_norm.mean()
    stdv = img_norm.std()
    strech_min = np.max([mean-scaling_params[0]*stdv, img_norm.min()])
    strech_max = np.min([mean+scaling_params[1]*stdv, img_norm.max()])
    img_norm[img_norm > strech_max] = strech_max
    img_norm[img_norm < strech_min] = strech_min
    img_norm = (img_norm-strech_min + 1e-8)/(strech_max-strech_min + 1e-8)
    img_norm = img_norm.clip(0,img_norm.max())

    # Haralick requires integer type data

    img_norm = (255*(img_norm/img_norm.max())).astype(np.uint8)

    # Haralick texture features as decribed in [1]. See [2] for original paper by Haralick et. al
    # [1] - https://mahotas.readthedocs.io/en/latest/api.html?highlight=mahotas.features.haralick
    # [2] - Haralick et. al. Textural features for image classification. IEEE Transactions on systems, man, and cybernetics, (6), 610-621.
    # Notice that a minimal number of pixels (512) is required for computing these features.

    number_of_voxels_required = 512

    if number_of_voxels >= number_of_voxels_required:
        ftextural = haralick(img, ignore_zeros=True, return_mean=True)

    for fid, fname in enumerate(
        [
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
            "texture_haralick_info_corr2",
        ]
    ):
        features[fname] = (
            np.nan if number_of_voxels <= number_of_voxels_required else ftextural[fid]
        )

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

    skel = skeletonize_3d(seg.astype(np.uint8))

    skel[skel > 0] = 1

    skel = np.pad(skel, 1, "constant")

    skel_degree = np.copy(skel)

    # Creating an image where the value of each pixel represents
    # its number of neighbors after skeletonization.

    z_pxl, y_pxl, x_pxl = np.where(skel > 0)

    nv = len(z_pxl)

    for x, y, z in zip(x_pxl, y_pxl, z_pxl):
        neigh = skel[z - 1 : z + 2, y - 1 : y + 2, x - 1 : x + 2]  # noqa
        skel_degree[z, y, x] = neigh.sum()

    nt = skel.sum()
    n0 = np.sum(skel_degree == (0 + 1))
    n1 = np.sum(skel_degree == (1 + 1))
    n2 = np.sum(skel_degree == (2 + 1))
    n3 = np.sum(skel_degree == (3 + 1))
    n4 = np.sum(skel_degree >= (4 + 1))

    # Average degree from <k> = Σ k x Pk

    if n2 != nt:
        average_degree = 0
        deg, Ndeg = np.unique(skel_degree.reshape(-1), return_counts=True)
        for k, n in zip(deg, Ndeg):
            if k != 2:
                average_degree = average_degree + k * (n / (nt - n2))
    else:
        average_degree = 1

    features["skeleton_voxels_number"] = nt
    features["skeleton_nodes_number"] = nt - n2
    features["skeleton_degree_mean"] = average_degree
    features["skeleton_edges_number"] = np.int(0.5 * (nt - n2) * average_degree)

    # Every pixel has to have at least one neighbor if the skeleton
    # contains more than a single pixel.

    if nt > 1:
        assert n0 == 0

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

    return features


def get_io_intensity_features(img, number_ops):

    """
        Applies erosion operation "number_ops" times
        to divide the image into 5 regions. We only
        care about the regions with id 2, 4 and 4 and
        we call them outer, mid and inner regions.
        These regions are defined by a sequence of
        erosion operations applyed to the input img.

                 1.  2.  3.    4.         5
        surface |-|----|----|----|----------------| center

                   <------------>
                    3 x number_ops

        :param seg: 3D 16-bit image (usually given by a multiplication
        of a gray scale image and its segmented version). The images
        contains a single connected component. Background has value 0
        and object of interest has value > 0.
        :number_ops: number of operations that define
        the width of regions 2, 3 and 4.

        :return: df - dictionary of features
    """

    features = {}

    if len(img.shape) != 3:
        raise ValueError("Incorrect dimensions: {}".format(img.shape))

    from skimage.morphology import binary_erosion

    img_bin = img.copy()
    img_bin[img_bin > 0] = 1

    z_pxl, y_pxl, x_pxl = np.where(img_bin > 0)

    # Sequence of erosion to create inner, mid and outer images

    img_rings = img_bin.copy()
    for r in range(4):
        if r > 0:
            for op in range(number_ops):
                img_bin = binary_erosion(img_bin)
        else:
            img_bin = binary_erosion(img_bin)
        img_rings += img_bin

    img_inner = img_rings.copy()
    img_inner = (img_inner==5).astype(np.uint8)

    img_mid = img_rings.copy()
    img_mid = (img_mid==4).astype(np.uint8)

    img_outer = img_rings.copy()
    img_outer = (img_outer==2).astype(np.uint8)

    # we do not compute anything if the inner region has size zero

    if img_inner.sum() > 0:

        # Center slice

        z, _, _ = np.nonzero(img)

        center_slice = np.int(z.mean())

        for img_region, region_name in zip([img_outer,img_mid,img_inner],["outer","mid","inner"]):

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

        for img_region, region_name in zip([img_outer,img_mid,img_inner],["outer","mid","inner"]):

            fea_name = "io_intensity_"+region_name+"_volume"
            features[fea_name] = np.nan

            fea_name = "io_intensity_"+region_name+"_mean"
            features[fea_name] = np.nan

            fea_name = "io_intensity_"+region_name+"_slice_mean"
            features[fea_name] = np.nan

    return features


def get_bright_spots_features(img):

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

    features = {}

    if len(img.shape) != 3:
        raise ValueError("Incorrect dimensions: {}".format(img.shape))

    from skimage.measure import label
    from skimage.filters import gaussian
    from skimage.morphology import extrema, binary_dilation

    SPOT_THRESHOLD = 0.85

    def norm_and_smooth(img_original):

        # Uses Jianxu's normalization and Gaussian smooth to
        # preprocess the input image. Parameters have default
        # values used in the segmentation toolkit.

        SMOOTH_SIGMA = 1.0
        SCALING_PARAMETERS = [0.5, 18]

        img_norm = img_original.copy()
        mean = img_norm.mean()
        stdv = img_norm.std()
        strech_min = np.max([mean-SCALING_PARAMETERS[0]*stdv, img_norm.min()])
        strech_max = np.min([mean+SCALING_PARAMETERS[1]*stdv, img_norm.max()])
        img_norm[img_norm > strech_max] = strech_max
        img_norm[img_norm < strech_min] = strech_min
        img_norm = (img_norm-strech_min + 1e-8)/(strech_max-strech_min + 1e-8)
        img_norm = gaussian(image=img_norm, sigma=SMOOTH_SIGMA)

        # img_norm may contain negative values

        return img_norm

    img_norm = norm_and_smooth(img_original=img)

    # Find maxima

    img_max = extrema.h_maxima(img_norm, h=0.1)
    z_pxl, y_pxl, x_pxl = np.nonzero(img_max)

    # Radius of the region cropped around each maxima. Final
    # patch will have size 2r+1  x 2r+1 x 3. Z direction is
    # not resampled.

    r = 11

    # For each maxima we crop a region and append their max
    # projection

    spots = []
    coords = []
    for n_id in range(len(x_pxl)):

        x = x_pxl[n_id]
        y = y_pxl[n_id]
        z = z_pxl[n_id]
        
        # check whether the neighborhod falls inside the image

        if (np.min([r,x,y,img.shape[2]-x-1,img.shape[1]-y-1]) == r) and (z > 2) and (z < img.shape[0]-3):
            
            # region round the bright spot

            img_spot = img[(z-2):(z+3),(y-r):(y+r+1),(x-r):(x+r+1)]
            
            # append in a list

            spots.append(img_spot.max(axis=0))

            # save coordinates

            coords.append((x,y,z))

        else:

            img_max[z,y,x] = 0

    # list to array

    spots = np.array(spots)
    spots.reshape(-1,2*r+1,2*r+1)

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

        if not (np.sum(dist<r) > 1):

            img_spot = spots[spot].copy()

            # normalize to max 1 and apply threshold

            max_intensity = img_spot.max()
            img_spot = img_spot / max_intensity
            img_spot_bin = (img_spot>SPOT_THRESHOLD).astype(np.uint8)

            # find the largest connected component that includes the central pixel

            img_spot_mask = label(img_spot_bin, connectivity=1)
            img_spot_mask = (img_spot_mask==img_spot_mask[r,r]).astype(np.uint8)

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

    return features

def get_dispersion_features(image1_seg, image2_seg, normalize=True, number_of_samples=50, repetitions=400):

    """
        Calculates the dispersion of objects in image1
        wrt objects in image2. Low values of dispersion
        indicate objects in image1 is more clustered
        than objects in image2.

        :param image1_seg: 3D binary image. Background has
        value 0 and object of interest has value > 0.
        :param image2_seg: 3D binary image. Background has
        value 0 and object of interest has value > 0.
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

    features = []

    if len(img.shape) != 3:
        raise ValueError("Incorrect dimensions: {}".format(img.shape))

    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment

    # Finding nonzero voxels

    coords = []
    coords.append(np.nonzero(image1_seg))
    coords.append(np.nonzero(image2_seg))
    
    for i in range(2):
        coords[i] = np.asarray(coords[i]).T
        #coords[i] = coords[i] + (np.random.rand(*coords[i].shape)-0.5)
        
    if coords[0].shape[0] > number_of_samples and coords[1].shape[0] > number_of_samples:

        centroid = []
        diameter = []
        for i in range(2):
            centroid.append(coords[i].mean(axis=0))
            diameter.append(np.sqrt(((coords[i]-centroid[i])**2).sum(axis=1)).max())

        # Finding the reference set as the one with largest
        # diameter, usually the cytoplasm

        reference = 0 if diameter[0] > diameter[1] else 1

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

            features.append({"cost": cost_str_mean,
                             "cost_local": cost_str_local_mean,
                             "dispersion": cost_str_std,
                             "dispersion_local": cost_str_local_std,
                             "type": "data"})
            features.append({"cost": cost_ctl_mean,
                             "cost_local": cost_ctl_local_mean,
                             "dispersion": cost_ctl_std,
                             "dispersion_local": cost_ctl_local_std,
                             "type": "control"})

        features = pd.DataFrame(features)
        features = features.groupby("type").mean()
        features = features.reset_index()

    else:

        features.append({"cost": np.nan,
                         "cost_local": np.nan,
                         "dispersion": np.nan,
                         "dispersion_local": np.nan,
                         "type": "data"})
        features.append({"cost": np.nan,
                         "cost_local": np.nan,
                         "dispersion": np.nan,
                         "dispersion_local": np.nan,
                         "type": "control"})
        features = pd.DataFrame(features)

    return features
