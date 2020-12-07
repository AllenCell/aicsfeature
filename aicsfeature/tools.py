import vtk
import pyshtools
import numpy as np
import pandas as pd
from vtk.util import *
from skimage import io as skio
from skimage import filters, morphology

def SimplifyMesh(polydata, target_reduction=0.9, n_smooth_iter=256):

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

def AlignPoints(x, y, z, from_axis=0, to_axis=0, align3d=False):
   
    from scipy import stats as spstats
    from sklearn import decomposition as skdecomp

    df = pd.DataFrame({'x':x,'y':y,'z':z})

    cartesian_axes = np.array([[1,0,0],[0,1,0],[0,0,1]])

    if align3d:

        eigenvecs = skdecomp.PCA(n_components=3).fit(df.values).components_

        theta = np.arccos(np.clip(np.dot(eigenvecs[from_axis], cartesian_axes[to_axis]), -1.0, 1.0))

        pivot = np.cross(eigenvecs[from_axis], cartesian_axes[to_axis])
        
        rot_mx = RotationMatrix(pivot, theta)
        
    else:

        eigenvecs = skdecomp.PCA(n_components=2).fit(df[['x','y']].values).components_

        theta = -np.arctan2(eigenvecs[0][1],eigenvecs[0][0])

        rot_mx = [[np.cos(theta),np.sin(-theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]]
        
    xyz_rot = np.dot(rot_mx, df.values.T).T

    # Dict to save alignment param

    params = dict()
    params['angle'] = theta

    # Post alignment processing:
    for ax, ax_name in zip([0,1],['xflip','yflip']):
        params[ax_name] = 1
        if np.corrcoef(xyz_rot[:,ax],xyz_rot[:,2])[0,1] < 0.0:
            xyz_rot[:,ax] *= -1
            params[ax_name] = -1

    if align3d:
        if np.corrcoef(xyz_rot[:,0],xyz_rot[:,1])[0,1] < 0.0:
            xyz_rot[:,2] *= -1

    fx, fy = 1, 1

    return fx*xyz_rot[:,0], fy*xyz_rot[:,1], xyz_rot[:,2], params

def GetPolyDataFromNumpy(volume, lcc=False, sigma=0, center=False, size_threshold=0):

    volume = np.swapaxes(volume,0,2)

    if lcc:

        volume = morphology.label(volume)

        counts = np.bincount(volume.reshape(-1))
        lcc = 1 + np.argmax(counts[1:])

        volume[volume!=lcc] = 0
        volume[volume==lcc] = 1

    if sigma:

        volume = volume.astype(np.float32)
        volume = filters.gaussian(volume,sigma=(sigma,sigma,sigma))
        volume[volume<1.0/np.exp(1.0)] = 0
        volume[volume>0] = 1
        volume = volume.astype(np.uint8)

    volume[[0,-1],:,:] = 0
    volume[:,[0,-1],:] = 0
    volume[:,:,[0,-1]] = 0
    volume = volume.astype(np.float32)

    if volume.sum() < size_threshold:
        return None, (None,None)

    img = vtk.vtkImageData()
    img.SetDimensions(volume.shape)

    volume = volume.transpose(2,1,0)
    volume_output = volume.copy()
    volume = volume.flatten()
    arr = numpy_support.numpy_to_vtk(volume, array_type=vtk.VTK_FLOAT)
    arr.SetName('Scalar')
    img.GetPointData().SetScalars(arr)

    cf = vtk.vtkContourFilter()
    cf.SetInputData(img)
    cf.SetValue(0, 0.5)
    cf.Update()

    polydata = cf.GetOutput()

    xo, yo, zo = 0, 0, 0
    if center:
        for i in range(polydata.GetNumberOfPoints()):
            x, y, z = polydata.GetPoints().GetPoint(i)
            xo += x
            yo += y
            zo += z
        xo /= polydata.GetNumberOfPoints()
        yo /= polydata.GetNumberOfPoints()
        zo /= polydata.GetNumberOfPoints()
        for i in range(polydata.GetNumberOfPoints()):
            x, y, z = polydata.GetPoints().GetPoint(i)
            polydata.GetPoints().SetPoint(i,x-xo,y-yo,z-zo)

    return cf.GetOutput(), (volume_output,(xo,yo,zo))

def GetReconstructionFromGrid(grid, cm=(0,0,0)):

    rec = vtk.vtkSphereSource()
    rec.SetPhiResolution(grid.shape[0]+2)
    rec.SetThetaResolution(grid.shape[1])
    rec.Update()
    rec = rec.GetOutput()

    n = rec.GetNumberOfPoints()
    res_lat = grid.shape[0]
    res_lon = grid.shape[1]

    grid = grid.T.flatten()

    # Coordinates

    for j, lon in enumerate(np.linspace(0, 2*np.pi, num=res_lon, endpoint=False)):
        for i, lat in enumerate(np.linspace(np.pi/(res_lat+1), np.pi, num=res_lat, endpoint=False)):
            theta = lat
            phi = lon - np.pi
            k = j * res_lat + i
            x = cm[0] + grid[k] * np.sin(theta)*np.cos(phi)
            y = cm[1] + grid[k] * np.sin(theta)*np.sin(phi)
            z = cm[2] + grid[k] * np.cos(theta)
            rec.GetPoints().SetPoint(k+2,x,y,z)
    north = grid[::res_lat].mean()
    south = grid[res_lat-1::res_lat].mean()
    rec.GetPoints().SetPoint(0,cm[0]+0,cm[1]+0,cm[2]+north)
    rec.GetPoints().SetPoint(1,cm[0]+0,cm[1]+0,cm[2]-south)

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(rec)
    normals.Update()

    return normals.GetOutput()

def GetReconstructionCoeffs(coeffs, l=None):

    if not l:
        l = coeffs.shape[-1]
    cfs = coeffs.copy()
    cfs[:,l:,:] = 0
    grid_rec = pyshtools.expand.MakeGridDH(cfs, sampling=2)
    rec = GetReconstructionFromGrid(grid_rec)
    return rec

def GetSHReconstructionFromXYZGrid(gridx, gridy, gridz, cm=(0,0,0)):

    rec = vtk.vtkSphereSource()
    rec.SetPhiResolution(gridx.shape[0]+2)
    rec.SetThetaResolution(gridx.shape[1])
    rec.Update()
    rec = rec.GetOutput()

    n = rec.GetNumberOfPoints()
    res_theta = gridx.shape[0]
    res_phi = gridx.shape[1]

    gridx = gridx.T.flatten()
    gridy = gridy.T.flatten()
    gridz = gridz.T.flatten()

    # Coordinates

    for j, phi in enumerate(np.linspace(0, 2*np.pi, num=res_phi, endpoint=False)):
        for i, theta in enumerate(np.linspace(np.pi/(res_theta+1), np.pi, num=res_theta, endpoint=False)):
            k = j * res_theta + i
            x = cm[0] + gridx[k]# * np.sin(theta)*np.cos(phi)
            y = cm[1] + gridy[k]# * np.sin(theta)*np.sin(phi)
            z = cm[2] + gridz[k]# * np.cos(theta)
            rec.GetPoints().SetPoint(k+2,x,y,z)
    northx = gridx[::res_theta].mean()
    northy = gridy[::res_theta].mean()
    northz = gridz[::res_theta].mean()
    southx = gridx[res_theta-1::res_theta].mean()
    southy = gridy[res_theta-1::res_theta].mean()
    southz = gridz[res_theta-1::res_theta].mean()
    rec.GetPoints().SetPoint(0,cm[0]+northx,cm[1]+northy,cm[2]+northz)
    rec.GetPoints().SetPoint(1,cm[0]+southx,cm[1]+southy,cm[2]+southz)
    rec.Modified()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(rec)
    normals.Update()

    return normals.GetOutput()

def GetSHReconstructionXYZCoeffs(coeffs_x,coeffs_y,coeffs_z,l=None):
    if not l:
        l = coeffs_x.shape[-1]
    cx = coeffs_x.copy()
    cx[:,l:,:] = 0
    cy = coeffs_y.copy()
    cy[:,l:,:] = 0
    cz = coeffs_z.copy()
    cz[:,l:,:] = 0
    grid_x_sh = pyshtools.expand.MakeGridDH(cx, sampling=2)
    grid_y_sh = pyshtools.expand.MakeGridDH(cy, sampling=2)
    grid_z_sh = pyshtools.expand.MakeGridDH(cz, sampling=2)
    rec = GetSHReconstructionFromXYZGrid(grid_x_sh, grid_y_sh, grid_z_sh)
    return rec