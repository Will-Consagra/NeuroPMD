import torch 
import numpy as np 

import os 
import pickle 
import scipy.io 

import sys 
from pathlib import Path

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
PATH2NPMD = os.path.join(str(parent_dir),"neuroPMD")
sys.path.append(PATH2NPMD)

from representation import PPLatentSphericalSiren
from utils import cart2sphere
from sph_harm_utils import clear_spherical_harmonics_cache

import matplotlib.pyplot as plt 
import argparse

#######################################################
####################Evaluation#########################
#######################################################

#### marginal plotting utility ####
def marg_plot(func_model, coords_surf_ix, coords_surf, chunk_size=10000):
	func_evals = []
	for i in range(coords_surf_ix.shape[0]):
		coords_six = coords_surf_ix[i,:]
		marg_coords_six = torch.from_numpy(np.column_stack((np.tile(coords_surf_ix[i,:], (coords_surf.shape[0],1)), coords_surf))).float()
		## batch eval to avoid overflow 
		f_chunks = []
		for j in range(marg_coords_six.shape[0]//chunk_size + 1):
			f_evals_chunk = func_model(marg_coords_six[chunk_size*j:(chunk_size*(j+1)),:].to(device))["model_out"]
			clear_spherical_harmonics_cache()
			f_chunks.append(f_evals_chunk.cpu().detach())
		fevals = torch.cat(f_chunks, dim=0)
		func_evals.append(fevals) 
	func_evals_tensor = torch.exp(torch.cat(func_evals,dim=1))
	func_evals_mean = func_evals_tensor.mean(dim=1).numpy()
	return func_evals_mean

def batch_eval_density(func_model, chunk_size = 10000):
	## compute normalization integral 
	log_norm_integral = torch.log((torch.exp(func_model(quadpoints_S2xS2)["model_out"])*quadweights_S2xS2.view(-1,1)).sum())
	Omege_chunks = []
	for i in range(Omega_X.shape[0]//chunk_size + 1):
		Omega_X_chunk = Omega_X[chunk_size*i:(chunk_size*(i+1)),:].float().to(device)
		intensity_evals_X_chunk = torch.exp(func_model(Omega_X_chunk)["model_out"] - log_norm_integral)
		Omege_chunks.append(intensity_evals_X_chunk.cpu().detach())
	density_evals_X = torch.cat(Omege_chunks, dim=0)
	## re-stack to grid for plotting 
	density_matrix = np.zeros((nverts, nverts)) 
	for idx in idx_map.keys():
		i, j = idx_map[idx]
		density_matrix[i,j] = density_evals_X[idx].cpu().detach().numpy().ravel()
	return density_matrix


## parse args
parser = argparse.ArgumentParser()

parser.add_argument('--device_num', type=int, required=True)
parser.add_argument('--max_degree', type=int, required=True)
parser.add_argument('--rank', type=int, required=True)
parser.add_argument('--depth', type=int, required=True)
parser.add_argument('--viz_dir', type=str, required=True, help="Path to the directory to visualization data")
parser.add_argument('--model_dir', type=str, default="", help="Path to pre-trained .pt model")

args = parser.parse_args()

# Access the arguments
device_num = args.device_num
max_degree = args.max_degree
rank = args.rank
depth = args.depth 
VIZDIR = args.viz_dir
MODELPT = args.model_dir

if not MODELPT:
	MODELPT = os.path.join(current_dir, "model", "fmodel_degree_%s_width_%s_depth_%s_w0_10_lam_0.001.pth" % (max_degree, rank, depth)) 

device = torch.device("cuda:%s"%device_num if torch.cuda.is_available() else "cpu")

## get quadrature expansion for evaluation purposes
with open(os.path.join(PATH2NPMD, "data", "Lebedev_degree19.pkl"), "rb") as pklfile:
	marginal_quadrature_object = pickle.load(pklfile)

quadpoints_S2 = torch.from_numpy(marginal_quadrature_object["quadpoints_S2"]).float()
quadweights_S2 = torch.from_numpy(marginal_quadrature_object["quadweights_S2"]).float()

## get spherical mapping of quadrature points 
quadpoints_S2_sph = cart2sphere(quadpoints_S2)
Q = quadpoints_S2.shape[0]

quadpoints_S2xS2 = torch.zeros(Q**2, 6).float()
quadpoints_S2xS2_sph = torch.zeros(Q**2, 4).float()
quadweights_S2xS2 = torch.zeros(Q**2).float()

qix = 0
for q in range(Q):
	for j in range(Q):
		quadpoints_S2xS2[qix, :3] = quadpoints_S2[q,:]
		quadpoints_S2xS2[qix, 3:] = quadpoints_S2[j,:]
		quadpoints_S2xS2_sph[qix, 0:2] = quadpoints_S2_sph[q, :]
		quadpoints_S2xS2_sph[qix, 2:] = quadpoints_S2_sph[j, :]
		quadweights_S2xS2[qix] = quadweights_S2[q]*quadweights_S2[j]
		qix += 1

quadpoints_S2xS2 = quadpoints_S2xS2.to(device)
quadweights_S2xS2 = quadweights_S2xS2.to(device)

## load trained model
width = rank ## width 
w0 = 10
out_channels = 1 ##univariate functional data 
torch.manual_seed(10)
func_model = PPLatentSphericalSiren(max_degree=max_degree, 
									rank=rank, 
									width = width, 
									depth = depth, 
									out_channels = out_channels, 
									w0=w0, 
									outermost_linear=True)
func_model.load_state_dict(torch.load(MODELPT, map_location=device))
func_model = func_model.to(device)

## get surfaces for plotting 
coords_lh_surf_lps = torch.load(os.path.join(current_dir, "surfaces", "lp_sph_surface_lps_verts.pt"))
coords_surf_frontalpole = torch.load(os.path.join(current_dir, "surfaces", "LH_frontalpole_verts.pt"))
coords_surf_medialorbitofrontal = torch.load(os.path.join(current_dir, "surfaces", "LH_medialorbitofrontal_verts.pt"))
coords_surf_temporalpole = torch.load(os.path.join(current_dir, "surfaces", "LH_temporalpole_verts.pt"))

## marginal means
func_evals_marg_mean_frontalpole = marg_plot(func_model, coords_surf_frontalpole, coords_lh_surf_lps, chunk_size=10000)
func_evals_marg_mean_medialorbitofrontal = marg_plot(func_model, coords_surf_medialorbitofrontal, coords_lh_surf_lps, chunk_size=10000)
func_evals_marg_mean_temporalpole = marg_plot(func_model, coords_surf_temporalpole, coords_lh_surf_lps, chunk_size=10000)
with open(os.path.join(VIZDIR, "f_evals_marg_mean.pkl"), "wb") as pklfile:
	pickle.dump((func_evals_marg_mean_frontalpole, func_evals_marg_mean_medialorbitofrontal, func_evals_marg_mean_temporalpole), pklfile)

## set up grid for evaluation (from SBCI)
X = np.load(os.path.join(PATH2NPMD, "data", "X0_grid.npy"))
nverts = X.shape[0]
Omega_X = np.zeros((nverts**2, 6))
idx = 0
idx_map = {}
for i in range(nverts): ## flatten product grid X x X 
	for j in range(nverts):
		Omega_X[idx,0:3] = X[i,:]
		Omega_X[idx,3:] = X[j,:]
		idx_map[idx] = (i,j)
		idx += 1

Omega_X = torch.from_numpy(Omega_X).float()

## compute LH-LH connectivity on ico4
C_SC_LH = batch_eval_density(func_model, chunk_size = 10000)
with open(os.path.join(VIZDIR, "CC_lh_ico4.npy"), "wb") as npfile:
	np.save(npfile, C_SC_LH)



