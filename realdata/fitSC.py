import torch 
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CyclicLR
import numpy as np 
import itertools 

from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import time 
import os 
import pickle 
import scipy.io 

import sys 
from pathlib import Path

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
PATH2NPMD = os.path.join(str(parent_dir),"neuroPMD")
sys.path.append(PATH2NPMD)

from data_object import PointPattern
from representation import PPLatentSphericalSiren, SphHarmTPLinModel
from utils import cart2sphere
from sph_harm_utils import clear_spherical_harmonics_cache

import matplotlib.pyplot as plt 
import argparse

## utility functions
def makeDir(dirpath):
	try:  
		os.mkdir(dirpath)  
	except OSError as error:  
		print(error)

def S2_get_tangent_basis(coords_sph):
	azimuthal = coords_sph[...,0:1]; inclination = coords_sph[...,1:];
	u1 = torch.cat((torch.cos(inclination)*torch.cos(azimuthal), 
					torch.cos(inclination)*torch.sin(azimuthal),
					-torch.sin(inclination)) ,dim=-1)
	u2 = torch.cat((-torch.sin(inclination)*torch.sin(azimuthal), 
					torch.sin(inclination)*torch.cos(azimuthal),
					torch.zeros(azimuthal.shape, device=azimuthal.device)) ,dim=-1)
	return u1, u2 

def UnifS2xS2(q):
	v1 = torch.randn(q,3); v2 = torch.randn(q,3);
	u1 = v1/torch.norm(v1, dim=1, p=2)[:,None];  u2 = v2/torch.norm(v2, dim=1, p=2)[:,None];  
	u = torch.cat((u1, u2), dim=1)
	return u 

def S2_S2_proj_matrix(batch_points):
	batch_points_sph = torch.cat((cart2sphere(batch_points[:,:3]), cart2sphere(batch_points[:,3:])), dim=1)
	tb_11, tb_12 = S2_get_tangent_basis(batch_points_sph[:, 0:2]); tb_21, tb_22 = S2_get_tangent_basis(batch_points_sph[:, 2:]);
	P_1 = tb_11.unsqueeze(2) * tb_12.unsqueeze(1); P_2 = tb_21.unsqueeze(2) * tb_22.unsqueeze(1);
	P = torch.cat((torch.cat((P_1, torch.zeros(batch_points.shape[0], 3, 3, device=batch_points.device)), dim=2),
				torch.cat((torch.zeros(batch_points.shape[0], 3, 3, device=batch_points.device), P_2), dim=2)), dim=1)
	return P 

def grad_euc(model, batch_points, eps):
	n, d = batch_points.shape
	gradients = torch.zeros_like(batch_points).to(device)
	for i in range(d):
		# Create perturbation vectors
		perturb = torch.zeros_like(batch_points)
		perturb[:, i] = eps
		# Compute function values at perturbed batch_points
		f_plus = model(batch_points + perturb)["model_out"]
		f_minus = model(batch_points - perturb)["model_out"]
		# Compute finite difference approximation of the gradient
		gradients[:, i] = ((f_plus - f_minus) / (2 * eps)).flatten()
	return gradients

def diag_hess_euc(model, batch_points, eps):
	n, d = batch_points.shape
	hess_diag = torch.zeros_like(batch_points)
	f_center = model(batch_points)["model_out"]
	for i in range(d):
		# Create perturbation vectors
		perturb = torch.zeros_like(batch_points)
		perturb[:, i] = eps
		# Compute function values at perturbed points
		f_plus = model((batch_points + perturb).to(device))["model_out"]
		f_minus = model((batch_points - perturb).to(device))["model_out"]
		# Compute finite difference approximation of the second derivative
		hess_diag[:,i] = ((f_plus - 2 * f_center + f_minus) / (eps ** 2)).flatten()
	return hess_diag

def hessian_euc(model, batch_points, eps):
	n, d = batch_points.shape
	hessians = torch.zeros(n, d, d, device=device)
	f_center = model(batch_points)["model_out"]
	perturb_i = torch.zeros_like(batch_points)
	perturb_j = torch.zeros_like(batch_points)
	for i in range(d):
		for j in range(d):
			perturb_i[:, i] = eps
			perturb_j[:, j] = eps
			if i == j:
				f_plus = model(batch_points + perturb_i)["model_out"]
				f_minus = model(batch_points - perturb_i)["model_out"]
				hessians[:, i, j] = ((f_plus - 2 * f_center + f_minus) / (eps ** 2)).flatten()
			else:
				f_plus_plus = model(batch_points + perturb_i + perturb_j)["model_out"]
				f_plus_minus = model(batch_points + perturb_i - perturb_j)["model_out"]
				f_minus_plus = model(batch_points - perturb_i + perturb_j)["model_out"]
				f_minus_minus = model(batch_points - perturb_i - perturb_j)["model_out"]
				hessians[:, i, j] = ((f_plus_plus - f_plus_minus - f_minus_plus + f_minus_minus) / (4 * eps ** 2)).flatten()
			perturb_i[:, i] = 0.0
			perturb_j[:, j] = 0.0
	return hessians

def gradient(model, batch_points, epsilon=1e-3):
	P = S2_S2_proj_matrix(batch_points)
	grad_euc_f = grad_euc(model, batch_points, eps=epsilon)
	grad_s2xs2_f = torch.bmm(P, grad_euc_f.unsqueeze(-1)).squeeze()
	return grad_s2xs2_f

def laplace(model, batch_points, epsilon=1e-3):
	P = S2_S2_proj_matrix(batch_points)
	hess_euc_f = hessian_euc(model, batch_points, eps=epsilon)
	lap_f = hessians = torch.zeros(batch_points.shape[0], device=device)
	for i in range(batch_points.shape[1]):
		P_i = P[:, i, :] 
		lap_f += torch.einsum('bi,bij,bj->b', P[:, i, :], hess_euc_f, P[:, i, :])
	return lap_f 

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

## parse args
parser = argparse.ArgumentParser()

parser.add_argument('--device_num', type=int, required=True)
parser.add_argument('--lambda_2', type=float, required=True)
parser.add_argument('--max_degree', type=int, required=True)
parser.add_argument('--rank', type=int, required=True)
parser.add_argument('--depth', type=int, required=True)
parser.add_argument('--cyclic', action='store_true')
parser.add_argument('--es', action='store_true')
parser.add_argument('--base_lr', type=float, default=1e-5)
parser.add_argument('--max_lr', type=float, default=1e-3)
parser.add_argument('--step_size_up', type=int, default=2000)
parser.add_argument('--data_dir', type=str, default="endpoints/10", help="Path to the data directory (/path/2/new_subjects/subject_name/)")
parser.add_argument('--model_dir', type=str, default="", help="Path to pre-trained .pt model")

args = parser.parse_args()

# Access the arguments
device_num = args.device_num
lambda_2 = args.lambda_2
max_degree = args.max_degree
rank = args.rank
depth = args.depth 
CYCLIC = args.cyclic
DATADIR = args.data_dir
PRETRAINED = args.model_dir
EARLYSTOP = args.es

print("Device:", device_num, "Lambda 2:", lambda_2, "max_degree:", max_degree, "rank:", rank, "depth:", depth, "cyclic:", CYCLIC, "eary_stop:", EARLYSTOP)

#### load data ####
endpoint_file = os.path.join(DATADIR, "LH__points_euc.pt")
points_tensor = torch.load(endpoint_file)

## select gpu if available
device = torch.device("cuda:%s"%device_num if torch.cuda.is_available() else "cpu")

## make reproducibles
torch.manual_seed(0)
np.random.seed(0)

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

## finite difference step size for computation of TV or roughness penalty, i.e. form the exp_{w_{j}}(\epsilon u_{{w_{j}}}^{(i)}), for i,j=1,2
epsilon = 1e-3

## normalize w.r.t. to same surface measure
Vol_OmegaXOmega = (4*torch.pi)**2

## create data object 
batch_frac = 2
## get small validation set 
n = points_tensor.size(0) 
train_prop = 0.95
n_train = int(train_prop * n)
batch_size = n_train//batch_frac
indices = torch.randperm(n) 
points_tensor_train = points_tensor[indices[:n_train],:]
points_tensor_test = points_tensor[indices[n_train:],:]
Ntest = points_tensor_test.shape[0]
O_train = PointPattern(points_tensor_train)
dataloader_train = DataLoader(O_train, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=0)

#######################################################
####################Configure network##################
#######################################################

#### Configure network ####

## set MC quadrature sizes 
Q1 = 10**5
Q2 = 10**4
width = rank ## width 
w0 = 10
out_channels = 1 ##univariate functional data 
lambda_1 = 0.0 ## TV prior regularization strength

torch.manual_seed(10)
func_model = PPLatentSphericalSiren(max_degree=max_degree, 
									rank=rank, 
									width = width, 
									depth = depth, 
									out_channels = out_channels, 
									w0=w0, 
									outermost_linear=True)
## optionally load a pre-trained model
if PRETRAINED:
	func_model.load_state_dict(torch.load(PRETRAINED))
	## todo: 
	## 1) re-initialize some of the parameters `reset_parameters()`
	## 2) make some of the parameters non-trainable w/ `nn.Parameter`

## put network on device 
func_model = func_model.to(device)

## perform gradient based optimization 
lr = 1e-4; num_epochs = 10000 
optim = torch.optim.Adam(params=func_model.parameters(),lr=lr)

## set learning rate schedule 
if CYCLIC:
	scheduler = CyclicLR(optim, base_lr=args.base_lr, max_lr=args.max_lr, step_size_up=args.step_size_up, mode="triangular2")

total_steps = 0

## early stopping
patience = 10 ## number of epochs to wait without significant improvement 
burn_in = 10 ## number of epochs to wait before checking
check_freq = 5 ## calculate criteria every `check_freq` epochs
min_delta = 0.01 ## minimum percent increase to be considered significant 
criteria = [np.Inf]
sig_improvement = []

## threshold for detecting exploding/vanishing gradients
some_high_threshold = 1e6; some_low_threshold = 1e-6; 

## make the required directories for the output 
expName = "fmodel_degree_%s_width_%s_depth_%s_w0_%s_lam_%s_batchfrac_%s_Vol_%s_Cyclic_%s"%(max_degree, width, depth, w0, lambda_2, CYCLIC)

MODEL_DIR = os.path.join(DATADIR, "models", expName)
FIG_DIR = os.path.join(DATADIR, "figures", expName)
MARG_DIR = os.path.join(DATADIR, "marg_means", expName)
  
makeDir(os.path.join(DATADIR, "models"))
makeDir(os.path.join(DATADIR, "figures"))
makeDir(os.path.join(DATADIR, "marg_means"))

makeDir(MODEL_DIR)
makeDir(FIG_DIR)
makeDir(MARG_DIR)

#######################################################
####################Estimate Weights###################
#######################################################

writer = SummaryWriter(os.path.join(DATADIR, "runs/%s"%expName))
for epoch in range(1, num_epochs+1):
	for step_i, batch_data_i in enumerate(dataloader_train):
		start_time = time.time()
		## step 1: form data term 
		n_i = batch_data_i.shape[0]
		batch_points_i = batch_data_i.to(device) 
		fmodel_data_i = func_model(batch_points_i)
		data_term_i = fmodel_data_i["model_out"].sum()
		clear_spherical_harmonics_cache()
		## step 2: form MC - normalization integral 
		batch_points_q1 = UnifS2xS2(Q1)
		model_quad_eval_i = func_model(batch_points_q1.to(device))["model_out"]
		norm_term_i = n_i*(Vol_OmegaXOmega/Q1)*torch.exp(model_quad_eval_i).sum()
		clear_spherical_harmonics_cache()
		## step 3: form MC - roughness penalty 
		batch_points_q2 = UnifS2xS2(Q2).to(device)
		#func_model_grad = gradient(func_model, batch_points_q2, epsilon=epsilon)
		#TV_prior = (Vol_OmegaXOmega/Q2)*((torch.abs(func_model_grad)).sum())
		#clear_spherical_harmonics_cache()
		func_model_lapl = laplace(func_model, batch_points_q2, epsilon=epsilon)
		Rough_prior = (Vol_OmegaXOmega/Q2)*((func_model_lapl**2).sum())
		clear_spherical_harmonics_cache()
		## step 4: perform gradient step 
		pp_likelihood_i = data_term_i - norm_term_i 
		f_loss_i = - pp_likelihood_i
		loss_i = f_loss_i + lambda_2*Rough_prior 
		optim.zero_grad()
		loss_i.backward()
		optim.step()
		if CYCLIC:
			scheduler.step()
		writer.add_scalar('Loss/total_loss', loss_i.item(), epoch * len(dataloader_train) + step_i)
		writer.add_scalar('Metrics/data_term', -data_term_i.item(), epoch * len(dataloader_train) + step_i)
		writer.add_scalar('Metrics/norm_integral', norm_term_i.item(), epoch * len(dataloader_train) + step_i)
		#writer.add_scalar('Metrics/TV_prior', TV_prior.item(), epoch * len(dataloader_train) + step_i)
		writer.add_scalar('Metrics/Rough_prior', Rough_prior.item(), epoch * len(dataloader_train) + step_i)
		for name, param in func_model.named_parameters():
			if param.requires_grad:
				writer.add_scalar(f"Metrics/{name}.grad", param.grad.norm().item(), epoch * len(dataloader_train) + step_i)
		for name, parameter in func_model.named_parameters():
			if parameter.grad is not None:
				grad_norm = parameter.grad.norm()
				if grad_norm > some_high_threshold:  # Checking for exploding
					print(f'Exploding Gradient-> Layer: {name} Grad Norm: {grad_norm}')
				if grad_norm < some_low_threshold:  # Checking for vanishing
					print(f'Vanishing Gradient-> Layer: {name} Grad Norm: {grad_norm}')
		#pbar.update(1)
		total_steps += 1
	print("Epoch %d, Loss %0.6f, iteration time %0.6f" % (epoch, loss_i, time.time() - start_time))
	if (not epoch % check_freq) and (patience >= epoch):
		######## estimate ISE criteria ########
		## 0) compute normalization integral 
		log_fhat = func_model(quadpoints_S2xS2)["model_out"]
		clear_spherical_harmonics_cache()
		log_norm_integral = torch.log((torch.exp(log_fhat)*quadweights_S2xS2.view(-1,1)).sum())
		fhat = torch.exp(log_fhat - log_norm_integral)
		## 1) compute \hat{f}
		fmodel_valid_i = torch.exp(func_model(points_tensor_test.to(device))["model_out"] - log_norm_integral)
		clear_spherical_harmonics_cache()
		## 2) compute \int \hat{f}^2 
		fmodel_l2_norm_quadrature_i = torch.square(fhat)*quadweights_S2xS2.view(-1,1)
		## 3) compute \int(f - \hat{f})^2 - \int f^2 
		approx_l2_error_i = fmodel_l2_norm_quadrature_i.sum() - ((2./Ntest)*Vol_OmegaXOmega*fmodel_valid_i.sum()) 
		## write results 
		writer.add_scalar('Metrics/max_value', fhat.max(), epoch)
		writer.add_scalar('Loss/ISE_estim', approx_l2_error_i.item(), epoch)
		writer.add_scalar('Loss/L2_energy', fmodel_l2_norm_quadrature_i.sum().item(), epoch)
		writer.add_scalar('Loss/L2_inner_prod', (2./Ntest)*Vol_OmegaXOmega*fmodel_valid_i.sum().item(), epoch)
		######## early stopping ########
		## store criteria 
		criteria.append(approx_l2_error_i.item())
		## do we see significant improvement
		if (criteria[-1]/criteria[-2] <= 1-min_delta):
			sig_improvement.append(True)
		## save model if it is the best one
		if criteria[-1] == min(criteria):
			torch.save(func_model.state_dict(), os.path.join(MODEL_DIR, "model_checkpoint_epoch_%s.pth"%(epoch,)) )
		## have we seen no significant improvement for over our `patience`?
		if (not any(sig_improvement[-patience:])) and EARLYSTOP:
			break

