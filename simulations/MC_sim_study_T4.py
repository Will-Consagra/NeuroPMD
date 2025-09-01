import torch 
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import itertools 
from functools import partial

from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import time 
import os 
import pickle 
import scipy.io
import argparse
import time 

from scipy.stats import qmc, pearsonr
from scipy.stats import vonmises_fisher

import sys 

PATH2NCP = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")), "neuroPMD"))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")), "optimization"))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")), "utils"))

from data_object import PointPattern
from representation import PPLatentToroidalSiren, TPFBLinearModel

from opt_algos import MAP_f, laplace, gradient 
from hyper_optim import CV_optim, validation_set_selection

from KDEs import Toroidal_KDE, kfold_BW_selector

import snef_prod 

import matplotlib.pyplot as plt 

def mc_TD_sampler(Q, D, scramble=True):
	sampler = qmc.Sobol(d=D, scramble=scramble)
	mc_int_points = torch.from_numpy(sampler.random_base2(int(np.log2(Q)))).float()
	mc_points = (2*torch.pi * mc_int_points - torch.pi)
	return mc_points

def batch_eval_quad(fun_model, chunk_size=1000):
	f_chunks = []; chunk_size = 1000
	with torch.no_grad():
		for j in range(quad_tensor.shape[0]//chunk_size + 1):
			f_evals_chunk = fun_model(quad_tensor[chunk_size*j:(chunk_size*(j+1)),:].to(device))["model_out"]
			f_chunks.append(f_evals_chunk.cpu().detach())
	log_fhat = torch.cat(f_chunks, dim=0)
	return log_fhat

## parse args 
parser = argparse.ArgumentParser()
parser.add_argument("mc_exp", type=int, help="Index for MC experiment.")
parser.add_argument("--data_dir", type=str, default="T4_data", help="Directory holding ground truth density function + observed data for each experiment.")
parser.add_argument("--out_dir", type=str, default="T4_data/T4_results", help="Directory to write results to")
parser.add_argument("--record", action="store_false", help="Write tensorboard information for covergence analysis?")

args = parser.parse_args()
mc_exp = args.mc_exp
data_dir = args.data_dir
out_dir = args.out_dir
RECORD = args.record

torch.manual_seed(10)
np.random.seed(10)

## select gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## load mixture model
data = scipy.io.loadmat(os.path.join(data_dir,"T4_MWN_aniso_%s.mat"%mc_exp))

# Extract the variables
points = data["points"]
sobol_points = data["sobolPoints"]
true_density_evals_unnorm = torch.from_numpy(data["mixture_evals"]).float()
true_density_evals = torch.from_numpy(data["mixture_evals_norm"]).float()
n = points.shape[0] ## number of samples

points_tensor = torch.from_numpy(points).float()
quad_tensor = torch.from_numpy(sobol_points).float().to(device)
dA = torch.tensor(1/quad_tensor.shape[0])

## standardize to unit circle 
norm_pdf = 2*np.pi

## diff-geometry of Torus 
D = 4

#### INIF Estimate ####
results_inif = {}

## network params 
max_freq = 20 ## degree of harmonic encoding 
rank = 128 
width = 128 ## width 
depth = 4 ## depth 
MC_Q = 2**16; MQ_Q_DO = 2**14
w0 = 10. ## to intialize sin activations 
out_channels = 1 ##univariate functional data 

## optimization params 
lr = 1e-4; num_epochs = 10000; HYPEROPT = "VALID"

## Run BO for hyperparameter optimization 
optimizer_ = partial(torch.optim.Adam)
field_model_ = partial(PPLatentToroidalSiren, D=D, outermost_linear=True, rank=rank, width=width, depth=depth, out_channels=out_channels, w0=w0, sep_enc=False)

## hyper parameters to be fixed
hyper_params = {"num_epochs":num_epochs,
				"rank":rank,
				"width": width, 
				"depth":depth,  
				"out_channels":out_channels,  
				"D":D,
				"max_freq":max_freq,
				"w0":w0,
				"lr":lr,
				"MC_Q":MC_Q,
				"MQ_Q_DO":MQ_Q_DO,
				"mc_sampler":mc_TD_sampler,
				"quad_tensor":quad_tensor,
				"cycle_lr":True,
				"base_lr":1e-5,
				"max_lr":1e-3,
				"step_size_up_frac":2,
				"cycle_mode":"triangular2",
				"outname":"",
}

train_prop = 0.95
batch_frac = 2
parameter_map_valid = [{"name": "lambda_2", "vals": [5e-4, 5e-3, 5e-2, 5e-1, 5e-0]},]; num_iter_track=100;

t1_inif = time.time()

#### Validation Set Selection ####
valid_set_select_results = validation_set_selection(points_tensor, field_model_, optimizer_, MAP_f, hyper_params, parameter_map_valid, 
						train_prop=train_prop, batch_frac=batch_frac, num_iter_track=num_iter_track)
ix_select = np.argmin([res["criteria"][-1] for res in valid_set_select_results]) ## last take last estimate for now, maybe introduce some smoothing here 
func_model = valid_set_select_results[ix_select]["train_modeled"]
best_parameters = valid_set_select_results[ix_select]["parameters"]
full_params = {**hyper_params, **best_parameters}

t2_inif = time.time() - t1_inif
## evaluations 
#log_fhat = func_model(quad_tensor)["model_out"]
log_fhat = batch_eval_quad(func_model)
log_norm_integral = torch.log(dA*(torch.exp(log_fhat)).sum())
density_hat_tensor = torch.exp(log_fhat - log_norm_integral).cpu().detach()
l2_loss = torch.mean((true_density_evals - density_hat_tensor)**2).item()
l2_loss_norm = torch.mean((true_density_evals - density_hat_tensor)**2).item()/torch.mean((true_density_evals**2)).item()
hellinger_loss = torch.mean((torch.sqrt(true_density_evals) - torch.sqrt(density_hat_tensor))**2).item()
l2_correlation = pearsonr(true_density_evals.cpu().detach().numpy().flatten(), density_hat_tensor.numpy().flatten())[0]
angular_error = np.arccos(pearsonr(np.sqrt(true_density_evals.cpu().detach().numpy().flatten()), np.sqrt(density_hat_tensor.numpy().flatten()))[0])
results_inif["AE"] = angular_error
results_inif["Normalized_L2_error"] = l2_loss_norm

## remove sampler + quad_tensor to avoid unpickling issues 
del full_params["mc_sampler"]
del full_params["quad_tensor"]
results_inif["hyper_params"] = full_params

with open(os.path.join(out_dir, "results_inif_%s.pkl"%mc_exp), "wb") as pklfile:
	pickle.dump(results_inif, pklfile)

#### KDE estimate w/ cross-validated b.w. estimate ####
results_kde = {}
t1_kde = time.time()
MC_Q_kde = 2**10; ## have to make quadrature grid smaller or the BW selection takes an unreasonable amount of time ...
sampler = qmc.Sobol(d=D, scramble=True)
quad_tensor_sobol = torch.from_numpy(sampler.random_base2(int(np.log2(MC_Q_kde)))).float()
dA = torch.tensor(1/quad_tensor_sobol.shape[0])
quad_weights = dA*torch.ones(quad_tensor_sobol.shape[0])
kappa_values = [1, 5, 10, 15, 20, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450] 
nfolds = 5
avg_ise_scores = kfold_BW_selector(kappa_values, points_tensor, quad_tensor_sobol, quad_weights, norm_pdf, nsplits=nfolds) ## avoid leave one out --> too slow!!
kappa_optim = kappa_values[np.argmin(avg_ise_scores)]
kappa_tensor_optim = torch.tensor([kappa_optim]*D)

product_kde_model = Toroidal_KDE(points_tensor, norm_term=norm_pdf)
## batch it up to fit in memory 
prod_kde_chunks = []; chunk_size = 1000
for i in range(quad_tensor.shape[0]//chunk_size):
    product_kde_model_hat_i = product_kde_model(quad_tensor.cpu().detach().numpy()[chunk_size*i:(chunk_size*(i+1)),:], kappa_tensor_optim)
    prod_kde_chunks.append(product_kde_model_hat_i.cpu().detach())
    print("Finished chunk ", i)

product_kde_model_hat = torch.cat(prod_kde_chunks)
product_kde_model_hat = product_kde_model_hat[:,None]
l2_loss_kde = torch.mean((true_density_evals - product_kde_model_hat)**2).item()
norm_l2_loss_kde = torch.mean((true_density_evals - product_kde_model_hat)**2).item()/torch.mean(true_density_evals**2).item()
hellinger_loss_kde = torch.mean((torch.sqrt(true_density_evals) - torch.sqrt(product_kde_model_hat))**2).item()
l2_correlation_kde = pearsonr(true_density_evals.flatten(), product_kde_model_hat.cpu().detach().numpy().flatten())[0]
angular_error_kde = np.arccos(pearsonr(np.sqrt(true_density_evals.flatten()), np.sqrt(product_kde_model_hat.cpu().detach().numpy().flatten()))[0])

t2_kde = time.time() - t1_kde

results_kde["time"] = t2_kde
results_kde["AE"] = angular_error_kde
results_kde["L2_error"] = l2_loss_kde
results_kde["Normalized_L2_error"] = norm_l2_loss_kde
results_kde["hell_loss"] = hellinger_loss_kde
results_kde["l2_corr"] = l2_correlation_kde
results_kde["kappa_optim"] = kappa_optim

with open(os.path.join(out_dir, "results_kde_%s.pkl"%mc_exp), "wb") as pklfile:
	pickle.dump(results_kde, pklfile)


##### TPB Estimate #####
results_tpb = {}

MC_Q = 2**10; MQ_Q_DO = 2**8
max_freq = 15 ## degree of harmonic encoding 
batch_frac = 20

optimizer_tpb_ = partial(torch.optim.Adam)
field_model_tpb_ = partial(TPFBLinearModel, D=D, max_freq=max_freq)

lr = 1e-5; num_epochs = 10000; HYPEROPT = "VALID"

## hyper parameters to be fixed
hyper_params_tpb = {"num_epochs":num_epochs,
                "D":D,
                "lr":lr,
                "MC_Q":MC_Q,
                "MQ_Q_DO":MQ_Q_DO,
                "mc_sampler":mc_TD_sampler,
                "cycle_lr":False,
                "base_lr":1e-5,
                "max_lr":1e-4,
                "step_size_up_frac":2,
                "cycle_mode":"triangular2",
                #"outname":"tpb",
                "outname":"",
                "VolOmega":norm_pdf**D
}

hyper_params_tpb["true_density_evals"] = true_density_evals

train_prop = 0.95
## random split 
n_train = int(train_prop * n)
batch_size = n//batch_frac
indices = torch.randperm(n) 
points_tensor_train = points_tensor[indices[:n_train],:]
points_tensor_test = points_tensor[indices[n_train:],:]
O_train = PointPattern(points_tensor_train)
dataloader = DataLoader(O_train, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=0)

approx_ise_crit = []
num_epochs =  hyper_params_tpb["num_epochs"]
MC_Q = hyper_params_tpb["MC_Q"]
MQ_Q_DO = hyper_params_tpb["MQ_Q_DO"]
D = hyper_params_tpb["D"]
lambda_1 = hyper_params_tpb.get("lambda_1")
mc_sampler = hyper_params_tpb["mc_sampler"]
outname = hyper_params_tpb["outname"]
Vol = hyper_params_tpb.get("Volume", 1)

TRACK_GT = True
TRACK_ISE = True
num_iter_track = 200

lambda_2 = 5e-0 ##pre-selected using validation set approach, too long to run selection for each MC experiment ....

## instantiate model and optimzier
func_model = field_model_tpb_().to(device)
optim = optimizer_tpb_(params=func_model.parameters(), lr = lr)
with tqdm(total=num_epochs) as pbar:
	for epoch in range(1, num_epochs+1):
		for step_i, batch_data_i in enumerate(dataloader):
			## form PP likelihood
			## put on gpu 
			batch_points = batch_data_i.to(device) 
			batch_size = batch_points.shape[0]
			## evaluate function represenation 
			fmodel_data_i = func_model(batch_points)
			## form PP likelihood
			data_term_i = fmodel_data_i["model_out"].sum()
			## quadrature integral 
			mc_points = mc_sampler(MC_Q, D).to(device)
			norm_term_i = (batch_size*(Vol/MC_Q)*torch.exp(func_model(mc_points)["model_out"])).sum()
			pp_likelihood_i = data_term_i - norm_term_i 
			f_loss_i = - pp_likelihood_i
			## smoothness prior via approximate TV 
			mc_diff_op_points_tensor = mc_sampler(MQ_Q_DO, D).to(device)
			#if DiffOpCalcDisc:
			#   grad_d = grad_discrete(func_model, mc_diff_op_points_tensor, eps)
			#   TV_prior = (Vol/MQ_Q_DO)*((torch.abs(grad_d)).sum())
			#   lap_d = laplace_discrete(func_model, mc_diff_op_points_tensor, eps)
			#   Rough_prior = (Vol/MQ_Q_DO)*(lap_d**2).sum()
			result = func_model.forward_wgrad(mc_diff_op_points_tensor)
			penalty_func = 0.0
			if lambda_1:
				TV_prior = (Vol/MQ_Q_DO)*((torch.abs(gradient(result["model_out"], result["model_in"]))).sum())
				penalty_func += lambda_1*TV_prior
			if lambda_2:
				Rough_prior = (Vol/MQ_Q_DO)*((laplace(result["model_out"], result["model_in"])**2).sum())
				penalty_func += lambda_2*Rough_prior
			## gradient step 
			loss_i = f_loss_i + penalty_func
			optim.zero_grad()
			loss_i.backward()
			optim.step()
		pbar.update(1)

## batch forward pass 
tpb_chunks = []; chunk_size = 1000
with torch.no_grad():
    for j in range(quad_tensor.shape[0]//chunk_size + 1):
        tpb_evals_chunk = func_model(quad_tensor[chunk_size*j:(chunk_size*(j+1)),:].to(device))["model_out"]
        tpb_chunks.append(tpb_evals_chunk.cpu().detach())


log_fhat = torch.cat(tpb_chunks, dim=0)
log_norm_integral = torch.log(dA*(torch.exp(log_fhat)).sum())
density_hat_tbp = torch.exp(log_fhat - log_norm_integral).cpu().detach()
l2_loss_tpb = torch.mean((true_density_evals - density_hat_tbp)**2).item()
norm_l2_loss_tpb = torch.mean((true_density_evals - density_hat_tbp)**2).item()/torch.mean(true_density_evals**2).item()
hellinger_loss_tpb = torch.mean((torch.sqrt(true_density_evals) - torch.sqrt(density_hat_tbp))**2).item()
l2_correlation_tpb = pearsonr(true_density_evals.cpu().detach().numpy().flatten(), density_hat_tbp.numpy().flatten())[0]
angular_error_tpb = np.arccos(pearsonr(np.sqrt(true_density_evals.cpu().detach().numpy().flatten()), np.sqrt(density_hat_tbp.numpy().flatten()))[0])
results_tpb["AE"] = angular_error_tpb
results_tpb["L2_error"] = l2_loss_tpb
results_tpb["Normalized_L2_error"] = norm_l2_loss_tpb
results_tpb["hell_loss"] = hellinger_loss_tpb
results_tpb["l2_corr"] = l2_correlation_tpb

with open(os.path.join(out_dir, "results_tpb_%s.pkl"%mc_exp), "wb") as pklfile:
	pickle.dump(results_tpb, pklfile)


##### pSNF Estimate #####
def extrincis(theta):
	x = torch.cos(theta)
	y = torch.sin(theta)
	coords = torch.stack([x, y], dim=-1)
	return coords

results_psnf = {}

quad_tensor_euc = torch.cat((extrincis(quad_tensor[:,0]), 
								extrincis(quad_tensor[:,1]),
								extrincis(quad_tensor[:,2]),
								extrincis(quad_tensor[:,3])), dim=1)
quad_tensor_euc = quad_tensor_euc.to(device)

## Product SNEF Model ##
domain = "sphere"
measure = "uniformsphere"
activation = "exp" 
encoding = "ident"

hidden_width = 257
output_width = 257
marg_dim = 2

snef_model = snef_prod.SquaredNNProd(D, domain, measure, activation, encoding, d=marg_dim, m=output_width, n=hidden_width)
snef_model = snef_model.to(device)

lr = 1e-5; num_epochs = 10000; 
optim = torch.optim.Adam(snef_model.parameters(), lr=lr)

## random split 
train_prop = 0.95; batch_frac = 2
n_train = int(train_prop * n)
batch_size = n_train//batch_frac
indices = torch.randperm(n) 

points_tensor_euc = torch.cat((extrincis(points_tensor[:,0]), 
								extrincis(points_tensor[:,1]),
								extrincis(points_tensor[:,2]),
								extrincis(points_tensor[:,3])), dim=1)
points_tensor_train = points_tensor_euc[indices[:n_train],:]
points_tensor_test = points_tensor_euc[indices[n_train:],:]


O_train = PointPattern(points_tensor_train)
dataloader = DataLoader(O_train, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=0)



TRACK_GT = False; num_iter_track = 100; outname="";

if outname:
    writer = SummaryWriter(outname)

t1_psnf = time.time()
with tqdm(total=len(dataloader)*num_epochs) as pbar:
	for epoch in range(num_epochs+1):
		for step_i, batch_data_i in enumerate(dataloader):
			## form PP likelihood
			## put on gpu 
			batch_points = batch_data_i.to(device) 
			batch_size = batch_points.shape[0]
			## form PP likelihood
			data_term = torch.sum(snef_model(batch_points, log_scale=True).T) #n/batch_size 
			## exact integral 
			norm_term = batch_size * snef_model.integrate(log_scale=False)
			pp_likelihood = data_term - norm_term 
			loss = - pp_likelihood
			## gradient step 
			optim.zero_grad()
			loss.backward()
			optim.step()
			pbar.update(1)
		if TRACK_GT and (not epoch % num_iter_track) and outname:
			log_fhat = snef_model(quad_tensor_euc, log_scale=True)
			log_norm_integral = snef_model.integrate(log_scale=True)
			density_hat_tensor = ((norm_pdf)**D) * torch.exp(log_fhat - log_norm_integral).cpu().detach().T
			l2_loss_norm = torch.mean((true_density_evals - density_hat_tensor)**2).item()/torch.mean((true_density_evals**2)).item()
			hellinger_loss = torch.mean((torch.sqrt(true_density_evals) - torch.sqrt(density_hat_tensor))**2).item()
			l2_correlation = pearsonr(true_density_evals.cpu().detach().numpy().flatten(), density_hat_tensor.numpy().flatten())[0]
			angular_error = np.arccos(pearsonr(np.sqrt(true_density_evals.cpu().detach().numpy().flatten()), np.sqrt(density_hat_tensor.numpy().flatten()))[0])
			print(epoch, "nISE", l2_loss_norm, "FR", angular_error, "Integrand", density_hat_tensor.mean().item(), "Var", torch.var(density_hat_tensor).item(), "Max", density_hat_tensor.max().item())
			writer.add_scalar('Metrics/max_value', density_hat_tensor.max().item(), epoch)
			writer.add_scalar('Loss/norm_L2_loss', l2_loss_norm, epoch)
			writer.add_scalar('Loss/Hellinger_Loss', hellinger_loss, epoch)
			writer.add_scalar('Loss/L2_corr', l2_correlation, epoch)
			writer.add_scalar('Loss/AE', angular_error, epoch)


t2_psnf = time.time() - t1_psnf

log_fhat = snef_model(quad_tensor_euc, log_scale=True)
log_norm_integral = snef_model.integrate(log_scale=True)
density_hat_tensor = ((norm_pdf)**D) * torch.exp(log_fhat - log_norm_integral).cpu().detach().T

angular_error_snef = np.arccos(pearsonr(np.sqrt(true_density_evals.cpu().detach().numpy().flatten()), np.sqrt(density_hat_tensor.numpy().flatten()))[0])
norm_l2_loss_snef = torch.mean((true_density_evals - density_hat_tensor)**2).item()/torch.mean((true_density_evals**2)).item()

results_snef = {}
results_snef["AE"] = angular_error_snef
results_snef["Normalized_L2_error"] = norm_l2_loss_snef

with open(os.path.join(out_dir, "results_snef_%s.pkl"%mc_exp), "wb") as pklfile:
	pickle.dump(results_snef, pklfile)


