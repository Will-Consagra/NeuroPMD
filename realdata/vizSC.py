import numpy as np 
import vtk
import torch 
import scipy.io

import pickle 
import os 
import sys 

import matplotlib.pyplot as plt 
import argparse

from pathlib import Path

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
PATH2NPMD = os.path.join(str(parent_dir),"neuroPMD")
sys.path.append(PATH2NPMD)


# Function to read the surface VTK file
def read_vtk(filename):
	reader = vtk.vtkPolyDataReader()
	reader.SetFileName(filename)
	reader.Update()
	return reader.GetOutput()

# Function to extract vertices from the VTK object
def get_vertices(polydata):
	points = polydata.GetPoints()
	vertices = []
	for i in range(points.GetNumberOfPoints()):
		point = points.GetPoint(i)
		vertices.append(point)
	return vertices

# Function to add scalar values to the vertices
def add_scalar_values_to_vtk(polydata, scalar_values):
	scalars = vtk.vtkFloatArray()
	scalars.SetName("Scalars")
	for value in scalar_values:
		scalars.InsertNextValue(value)
	polydata.GetPointData().SetScalars(scalars)

# Function to write the modified VTK file
def write_vtk(polydata, filename):
	writer = vtk.vtkPolyDataWriter()
	writer.SetFileName(filename)
	writer.SetInputData(polydata)
	writer.Write()

# Function to convert numpy array of points to VTK points
def numpy_to_vtk_points(points):
	vtk_points = vtk.vtkPoints()
	for point in points:
		vtk_points.InsertNextPoint(point)
	return vtk_points

# Function to create a vtkPolyData object from numpy array of points
def create_vtk_polydata_from_points(points):
	polydata = vtk.vtkPolyData()
	polydata.SetPoints(numpy_to_vtk_points(points))
	# Create a vertex cell array to store the points
	vertices = vtk.vtkCellArray()
	for i in range(points.shape[0]):
		vertices.InsertNextCell(1)
		vertices.InsertCellPoint(i)
	polydata.SetVerts(vertices)
	return polydata

def chunked_arccos(tensor_1, coords, chunk_size):
    num_rows = tensor_1.shape[0]
    result = []
    for start_idx in range(0, num_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, num_rows)
        chunk = tensor_1[start_idx:end_idx]
        # Perform the matrix multiplication for the current chunk
        chunk_result = torch.arccos(torch.clip(chunk @ coords.T, -1, 1))
        # Append the result for this chunk to the final result list
        result.append(chunk_result)
    # Concatenate all the chunk results back into a single tensor
    return torch.cat(result, dim=0)


## parse args
parser = argparse.ArgumentParser()

parser.add_argument('--marg_surf_f', type=str, required=True)
parser.add_argument('--sc_f', type=str, required=True)
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--ep_f', type=str, required=True)

args = parser.parse_args()

# Access the arguments
MSFILE = args.marg_surf_f
SCFILE = args.sc_f
OUTDIR = args.out_dir
endpoint_file = args.ep_f

## get surface files 
fname_surf = os.path.join(current_dir, "surfaces", "vtks", "lh_white_avg_lps.vtk")
fname_sph = os.path.join(current_dir, "surfaces", "vtks", "lh_sphere_avg_lps.vtk")

if not os.path.exists(OUTDIR):
	os.makedirs(OUTDIR)

with open(MSFILE, "rb") as pklfile:
	func_evals_marg_mean_frontalpole, func_evals_marg_mean_medialorbitofrontal, func_evals_marg_mean_temporalpole = pickle.load(pklfile)


## surface plots 
surface_polydata = read_vtk(fname_surf)
add_scalar_values_to_vtk(surface_polydata, np.log(10e7*func_evals_marg_mean_frontalpole +1))
write_vtk(surface_polydata, os.path.join(OUTDIR, "frontal_pole.vtk"))

surface_polydata = read_vtk(fname_surf)
add_scalar_values_to_vtk(surface_polydata, np.log(10e7*func_evals_marg_mean_medialorbitofrontal +1))
write_vtk(surface_polydata, os.path.join(OUTDIR, "medialorbitofrontal_pole.vtk"))

surface_polydata = read_vtk(fname_surf)
add_scalar_values_to_vtk(surface_polydata, np.log(10e7*func_evals_marg_mean_temporalpole +1))
write_vtk(surface_polydata, os.path.join(OUTDIR, "temporal_pole.vtk"))

## spherical plots 
s2_polydata = read_vtk(fname_sph)
add_scalar_values_to_vtk(s2_polydata, np.log(10e7*func_evals_marg_mean_frontalpole +1))
write_vtk(s2_polydata, os.path.join(OUTDIR, "frontal_pole_S2.vtk"))

s2_polydata = read_vtk(fname_sph)
add_scalar_values_to_vtk(s2_polydata, np.log(10e7*func_evals_marg_mean_medialorbitofrontal +1))
write_vtk(s2_polydata, os.path.join(OUTDIR, "medialorbitofrontal_pole_S2.vtk"))

s2_polydata = read_vtk(fname_sph)
add_scalar_values_to_vtk(s2_polydata, np.log(10e7*func_evals_marg_mean_temporalpole +1))
write_vtk(s2_polydata, os.path.join(OUTDIR, "temporal_pole_S2.vtk"))

#### load data ####

## get spherical coordinates 
s2_polydata = read_vtk(fname_sph)
vertices = np.array(get_vertices(s2_polydata))
coordinates = vertices/np.linalg.norm(vertices, axis=1)[:,None]
nverts = vertices.shape[0]

points_tensor = torch.load(endpoint_file)
points_tensor_numpy = points_tensor.cpu().detach().numpy()

## get index set for closest endpoint for point-set 
surface_point_index = np.zeros((points_tensor.shape[0], 2), dtype=int)
for i in range(points_tensor.shape[0]):
	min_surf_ix1 = np.argmin(np.arccos(np.clip(coordinates @ points_tensor_numpy[i,:3],-1,1)))
	min_surf_ix2 = np.argmin(np.arccos(np.clip(coordinates @ points_tensor_numpy[i,3:],-1,1)))
	surface_point_index[i,0] = min_surf_ix1
	surface_point_index[i,1] = min_surf_ix2

surface_polydata = read_vtk(fname_surf)
surface_vertices = np.array(get_vertices(surface_polydata))

## Plot Endpoint ROIs on SURFACE 
for fi in range(1,4):
	if fi == 1:
		surf_ix = scipy.io.loadmat(os.path.join(current_dir, "surfaces", "indices", "LH_temporalpole.mat"))["surf_ix"].ravel(); func_evals_mean = np.log(10e7*func_evals_marg_mean_temporalpole+1); suffix = "LH_temporalpole"
	elif fi == 2:
		surf_ix = scipy.io.loadmat(os.path.join(current_dir, "surfaces", "indices", "LH_medialorbitofrontal.mat"))["surf_ix"].ravel(); func_evals_mean = np.log(10e7*func_evals_marg_mean_medialorbitofrontal+1); suffix = "LH_medialorbitofrontal"
	elif fi == 3:
		surf_ix = scipy.io.loadmat(os.path.join(current_dir, "surfaces", "indices", "LH_frontalpole.mat"))["surf_ix"].ravel(); func_evals_mean = np.log(10e7*func_evals_marg_mean_frontalpole+1); suffix = "LH_frontalpole"
	## get starting points in the surf_ix 
	surf_ix = surf_ix - 1
	starting_points = []; ending_points = []
	s2_starting_points = []; s2_ending_points = []
	for i in range(points_tensor.shape[0]):
		if surface_point_index[i,0] in surf_ix:
			starting_points.append(surface_vertices[surface_point_index[i,0]])
			ending_points.append(surface_vertices[surface_point_index[i,1]])
			s2_starting_points.append(vertices[surface_point_index[i,0]])
			s2_ending_points.append(vertices[surface_point_index[i,1]])
	starting_points_array = np.array(starting_points)
	ending_points_array = np.array(ending_points)
	s2_starting_points_array = np.array(s2_starting_points)
	s2_ending_points_array = np.array(s2_ending_points)
	## write endpoints 
	start_points_polydata = create_vtk_polydata_from_points(starting_points_array)
	write_vtk(start_points_polydata, os.path.join(OUTDIR, "ep1_%s.vtk"%(suffix,)))
	end_points_polydata = create_vtk_polydata_from_points(ending_points_array)
	write_vtk(end_points_polydata, os.path.join(OUTDIR, "ep2_%s.vtk"%(suffix,)))
	S2_start_points_polydata = create_vtk_polydata_from_points(s2_starting_points_array)
	write_vtk(S2_start_points_polydata, os.path.join(OUTDIR, "S2_ep1_%s.vtk"%(suffix,)))
	S2_end_points_polydata = create_vtk_polydata_from_points(s2_ending_points_array)
	write_vtk(S2_end_points_polydata, os.path.join(OUTDIR, "S2_ep2_%s.vtk"%(suffix,)))

## Plot Raw Endpoints from ROIs on SPHERE 
for fi in range(1,4):
	if fi == 1:
		surf_ix = scipy.io.loadmat(os.path.join(current_dir, "surfaces", "indices", "LH_frontalpole.mat"))["surf_ix"].ravel(); func_evals_mean = np.log(10e7*func_evals_marg_mean_frontalpole+1); suffix = "LH_frontalpole"
	elif fi == 2:
		surf_ix = scipy.io.loadmat(os.path.join(current_dir, "surfaces", "indices", "LH_medialorbitofrontal.mat"))["surf_ix"].ravel(); func_evals_mean = np.log(10e7*func_evals_marg_mean_medialorbitofrontal+1); suffix = "LH_medialorbitofrontal"
	elif fi == 3:
		surf_ix = scipy.io.loadmat(os.path.join(current_dir, "surfaces", "indices", "LH_temporalpole.mat"))["surf_ix"].ravel(); func_evals_mean = np.log(10e7*func_evals_marg_mean_temporalpole+1); suffix = "LH_temporalpole"
	surf_ix = surf_ix - 1
	coords_surf_ix = coordinates[surf_ix,:]
	dist_thresh = 0.02
	ep1_2_roi_dist = torch.arccos(points_tensor[:,:3] @ coords_surf_ix.T)
	ep1_2_roi_dist_min, _ = ep1_2_roi_dist.min(axis=1)
	ix_2_include = ep1_2_roi_dist_min < dist_thresh
	ep2_tensor = points_tensor[ix_2_include,3:]
	## write endpoints 
	points_polydata = create_vtk_polydata_from_points(100*ep2_tensor.cpu().detach().numpy())
	write_vtk(points_polydata, os.path.join(OUTDIR, "raw_endpoints_%s.vtk"%(suffix,)))


##plot SC
C_SC_LH = np.load(SCFILE)

plt.imshow(np.log(10e07 * C_SC_LH+1), cmap="viridis", interpolation="nearest")
plt.colorbar()  # Add color scale
plt.title("LH-SC")
plt.savefig(os.path.join(OUTDIR, "SC.png"))



