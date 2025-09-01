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

def mc_torus_sampler(Q, D, scramble=True):
	sampler = qmc.Sobol(d=D, scramble=scramble)
	mc_int_points = torch.from_numpy(sampler.random_base2(int(np.log2(Q)))).float()
	mc_points = (2*torch.pi * mc_int_points - torch.pi)
	return mc_points

#### Simulate mixture model on T^2 ####

## parse args 
parser = argparse.ArgumentParser()
parser.add_argument("mc_exp", type=int, help="Index for MC experiment.")
parser.add_argument("--data_dir", type=str, default="T2_data", help="Directory holding ground truth density function + observed data for each experiment.")
parser.add_argument("--out_dir", type=str, default="T2_data/T2_results", help="Directory to write results to")
parser.add_argument("--record", action="store_false", help="Write tensorboard information for covergence analysis?")

args = parser.parse_args()
mc_exp = args.mc_exp
data_dir = args.data_dir
out_dir = args.out_dir
RECORD = args.record

torch.manual_seed(10)
np.random.seed(10)

## select gpu if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

## load mixture model
data = scipy.io.loadmat(os.path.join(data_dir,"T2_MWN_aniso_%s.mat"%mc_exp))

# Extract the variables
points = data["points"]
true_density_evals_unnorm = data["mixture_evals"].T
true_density_matrix_unnorm = data["mixture_evals_2D"].T
n = points.shape[0] ## number of samples

points_tensor = torch.from_numpy(points).float()

## standardize to unit circle 
norm_pdf = 2*np.pi

## set up integration 
l_bounds = [0, 0]; u_bounds = [2*np.pi, 2*np.pi];
# Generate a grid of points within the integration limits

Qgrid = 100
S11, S12 = torch.meshgrid(torch.linspace(0, 2*torch.pi, Qgrid), 
						  torch.linspace(0, 2*torch.pi, Qgrid), 
						  indexing="ij")
quad_tensor = torch.stack((S11.reshape(-1), S12.reshape(-1)), dim=1).float().to(device)
quad_tensor_numpy = quad_tensor.cpu().detach().numpy()
dA = torch.tensor(1/quad_tensor_numpy.shape[0])

gt_norm_term = dA * true_density_evals_unnorm.sum()
true_density_evals = true_density_evals_unnorm/(gt_norm_term)
true_density_matrix = true_density_matrix_unnorm/gt_norm_term

## diff-geometry of Torus 
D = 2

#### INIF Estimate ####
## network params 
max_freq = 20 ## degree of harmonic encoding 
rank = 128 
width = 128 ## width 
depth = 3 ## depth 
MC_Q = 2**10; MQ_Q_DO = 2**10 ## MC-integration sizes
w0 = 10. ## to intialize sin activations 
out_channels = 1 ##univariate functional data 

## optimization params 
lr = 1e-4; num_epochs = 10000; 

## Run BO for hyperparameter optimization 
optimizer_ = partial(torch.optim.Adam)
field_model_ = partial(PPLatentToroidalSiren, D=D, outermost_linear=True)

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
                "mc_sampler":mc_torus_sampler,
                "quad_tensor":quad_tensor,
                "cycle_lr":False,
                "outname":"",
}

results_inif = {}

Nexperiments = 10
train_prop = 0.9
batch_frac = 1

parameter_map_bo = [{"name": "lambda_2", "type": "range", "bounds": [1e-4,1e1], "log_scale": True},]
parameter_map_cv = [{"name": "lambda_2", "vals": [1e-4, 1e-3, 1e-2, 1e-1, 1]},]; nfolds_cv = 5 
parameter_map_valid = [{"name": "lambda_2", "vals": [5e-4, 5e-3, 5e-2, 5e-1, 5e-0]},]; num_iter_track=100;

t1_inif = time.time()

#### Validation Set Selection ####
valid_set_select_results = validation_set_selection(points_tensor, field_model_, optimizer_, MAP_f, hyper_params, parameter_map_valid, train_prop=train_prop, batch_frac=batch_frac, num_iter_track=num_iter_track)
ix_select = np.argmin([res["criteria"][-1] for res in valid_set_select_results]) ## last take last estimate for now, maybe introduce some smoothing here 
func_model = valid_set_select_results[ix_select]["train_modeled"]
best_parameters = valid_set_select_results[ix_select]["parameters"]
full_params = {**hyper_params, **best_parameters}

t2_inif = time.time() - t1_inif
## evaluation 
log_fhat = func_model.forward_wgrad(quad_tensor)["model_out"]
log_norm_integral = torch.log(dA*(torch.exp(func_model.forward_wgrad(quad_tensor)["model_out"])).sum())
density_hat = torch.exp(log_fhat - log_norm_integral)
density_hat_inif_tensor = density_hat[...,0].reshape(Qgrid,Qgrid).cpu().detach().numpy()
l2_loss_inif = (np.linalg.norm(true_density_matrix - density_hat_inif_tensor, ord="fro")**2)*(1./(Qgrid**2))
norm_l2_loss_inif = l2_loss_inif/((np.linalg.norm(true_density_matrix, ord="fro")**2)*(1./(Qgrid**2)))
hellinger_loss_inif = (np.linalg.norm(np.sqrt(true_density_matrix) - np.sqrt(density_hat_inif_tensor), ord="fro")**2)*(1./(Qgrid**2))
l2_correlation_inif = pearsonr(true_density_matrix.flatten(), density_hat_inif_tensor.flatten())[0]
angular_error_inif = np.arccos(pearsonr(np.sqrt(true_density_matrix.flatten()), np.sqrt(density_hat_inif_tensor.flatten()))[0])

results_inif["time"] = t2_inif
results_inif["AE"] = angular_error_inif
results_inif["L2_error"] = l2_loss_inif
results_inif["Normalized_L2_error"] = norm_l2_loss_inif
results_inif["hell_loss"] = hellinger_loss_inif
results_inif["l2_corr"] = l2_correlation_inif

## remove sampler + quad_tensor to avoid unpickling issues 
del full_params["mc_sampler"]
del full_params["quad_tensor"]
results_inif["hyper_params"] = full_params


with open(os.path.join(out_dir, "results_inif_%s.pkl"%mc_exp), "wb") as pklfile:
	pickle.dump(results_inif, pklfile)

#### KDE estimate w/ cross-validated b.w. estimate ####
results_kde = {}
t1_kde = time.time()
quad_weights = dA*torch.ones(quad_tensor.shape[0])
kappa_values = [175, 200, 225, 230, 240, 250, 260, 270, 275, 300, 325, 350, 375, 400] 
nfolds = 5
avg_ise_scores = kfold_BW_selector(kappa_values, points_tensor, quad_tensor.cpu().detach(), quad_weights, norm_pdf, nsplits=nfolds) ## avoid leave one out --> too slow!!
kappa_optim = kappa_values[np.argmin(avg_ise_scores)]
kappa_tensor_optim = torch.tensor([kappa_optim]*D)

product_kde_model = Toroidal_KDE(points_tensor, norm_term=norm_pdf)
product_kde_model_hat = product_kde_model(quad_tensor.cpu().detach(), kappa_tensor_optim)

t2_kde = time.time() - t1_kde
l2_loss_kde = (np.linalg.norm(true_density_matrix - product_kde_model_hat.cpu().detach().numpy().reshape(Qgrid,Qgrid), ord="fro")**2)*(1./(Qgrid**2))
norm_l2_loss_kde = l2_loss_kde/((np.linalg.norm(true_density_matrix, ord="fro")**2)*(1./(Qgrid**2)))
hellinger_loss_kde = (np.linalg.norm(np.sqrt(true_density_matrix) - np.sqrt(product_kde_model_hat.cpu().detach().numpy().reshape(Qgrid,Qgrid)), ord="fro")**2)*(1./(Qgrid**2))
l2_correlation_kde = pearsonr(true_density_matrix.flatten(), product_kde_model_hat.cpu().detach().numpy().flatten())[0]
angular_error_kde = np.arccos(pearsonr(np.sqrt(true_density_matrix.flatten()), np.sqrt(product_kde_model_hat.cpu().detach().numpy().flatten()))[0])

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

max_freq = 20; 

optimizer_tpb_ = partial(torch.optim.Adam)
field_model_tpb_ = partial(TPFBLinearModel, D=D, max_freq=max_freq)

## hyper parameters to be fixed
hyper_params_tpb = {"num_epochs":num_epochs,
                "D":D,
                "lr":lr,
                "MC_Q":MC_Q,
                "MQ_Q_DO":MQ_Q_DO,
                "mc_sampler":mc_torus_sampler,
                "quad_tensor":quad_tensor,
                "outname":"",
                "VolOmega":norm_pdf**D
}

hyper_params_tpb["true_density_evals"] = true_density_evals

train_prop = 0.9; batch_frac = 1
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

TRACK_GT = False
TRACK_ISE = True
num_iter_track = 200

## Run SGA scheme for each lambda and select best result via Approx. ISE criteria on valid set 
for lambda_2 in  [5e-7, 5e-5, 5e-4, 5e-3, 5e-2, 5e-1, 5e-0]:
    if outname:
        writer = SummaryWriter("T2_LM/" + outname + "lam2_%s"%lambda_2)  
    ## instantiate model and optimzier
    func_model = field_model_tpb_().to(device)
    optim = optimizer_tpb_(params=func_model.parameters(), lr = lr)
    results_tpb[lambda_2] = {}
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
                if outname:
                    writer.add_scalar('Loss/total_loss', loss_i.item(), epoch * len(dataloader) + step_i)
                    writer.add_scalar('Metrics/data_term', -data_term_i.item(), epoch * len(dataloader) + step_i)
                    writer.add_scalar('Metrics/norm_integral', norm_term_i.item(), epoch * len(dataloader) + step_i)
                    if lambda_1:
                        writer.add_scalar('Metrics/TV_prior', TV_prior.item(), epoch * len(dataloader) + step_i)
                    if lambda_2:
                        writer.add_scalar('Metrics/Roughness_prior', Rough_prior.item(), epoch * len(dataloader) + step_i)
                    for name, param in func_model.named_parameters():
                        if param.requires_grad:
                            writer.add_scalar(f"Metrics/{name}.grad", param.grad.norm().item(), epoch * len(dataloader) + step_i)
            pbar.update(1)
            if TRACK_GT and (not epoch % num_iter_track) and outname:
                log_fhat = func_model(quad_tensor)["model_out"]
                log_norm_integral = torch.log(dA*(torch.exp(log_fhat)).sum())
                density_hat_tensor = torch.exp(log_fhat - log_norm_integral).cpu().detach().T
                l2_loss = torch.mean((true_density_evals - density_hat_tensor)**2).item()
                l2_loss_norm = torch.mean((true_density_evals - density_hat_tensor)**2).item()/torch.mean((true_density_evals**2)).item()
                hellinger_loss = torch.mean((torch.sqrt(true_density_evals) - torch.sqrt(density_hat_tensor))**2).item()
                l2_correlation = pearsonr(true_density_evals.cpu().detach().numpy().flatten(), density_hat_tensor.numpy().flatten())[0]
                angular_error = np.arccos(pearsonr(np.sqrt(true_density_evals.cpu().detach().numpy().flatten()), np.sqrt(density_hat_tensor.numpy().flatten()))[0])
                writer.add_scalar('Metrics/max_value', density_hat_tensor.max().item(), epoch)
                writer.add_scalar('Loss/L2_loss', l2_loss, epoch)
                writer.add_scalar('Loss/Hellinger_Loss', hellinger_loss, epoch)
                writer.add_scalar('Loss/L2_corr', l2_correlation, epoch)
                writer.add_scalar('Loss/AE', angular_error, epoch)
            if TRACK_ISE and (not epoch % num_iter_track):
                Ntest = points_tensor_test.size(0)
                model_quad_eval_i = func_model(quad_tensor)["model_out"]
                log_norm_integral = torch.log(dA*(torch.exp(model_quad_eval_i)).sum()).to("cpu")
                ## 1) compute \hat{f} on validation sets
                log_fhat = func_model(points_tensor_test.to(device))["model_out"].to("cpu")
                fmodel_valid_i = torch.exp(log_fhat - log_norm_integral)
                ## 2) compute \int \hat{f}^2 using quadrature points
                fmodel_l2_norm_quadrature_i = torch.square(torch.exp(model_quad_eval_i - log_norm_integral))*dA
                ## 3) compute [\int (f-\hat{f})^2] - \int f^2 = \int \hat{f}^2 - 2\int f\hat{f}
                approx_l2_error_i = fmodel_l2_norm_quadrature_i.sum() - (2./Ntest)*fmodel_valid_i.sum() 
                if outname:
                    writer.add_scalar('Loss/Approx_ISE', approx_l2_error_i.item(), epoch)
                    writer.add_scalar('Loss/L2_energy', fmodel_l2_norm_quadrature_i.sum().item(), epoch)
                    writer.add_scalar('Loss/L2_inner_prod', (2./Ntest)*fmodel_valid_i.sum().item(), epoch)
                approx_ise_crit.append(approx_l2_error_i.item())
        log_fhat = func_model.forward_wgrad(quad_tensor)["model_out"]
        log_norm_integral = torch.log(dA*(torch.exp(func_model.forward_wgrad(quad_tensor)["model_out"])).sum())
        density_hat_tpb = torch.exp(log_fhat - log_norm_integral)
        density_hat_tbp = density_hat_tpb[...,0].reshape(Qgrid,Qgrid).cpu().detach().numpy()
        l2_loss_tpb = (np.linalg.norm(true_density_matrix - density_hat_tbp, ord="fro")**2)*(1./(Qgrid**2))
        norm_l2_loss_tpb = l2_loss_tpb/((np.linalg.norm(true_density_matrix, ord="fro")**2)*(1./(Qgrid**2)))
        hellinger_loss_tpb = (np.linalg.norm(np.sqrt(true_density_matrix) - np.sqrt(density_hat_tbp), ord="fro")**2)*(1./(Qgrid**2))
        l2_correlation_tpb = pearsonr(true_density_matrix.flatten(), density_hat_tbp.flatten())[0]
        angular_error_tpb = np.arccos(pearsonr(np.sqrt(true_density_matrix.flatten()), np.sqrt(density_hat_tbp.flatten()))[0])
        results_tpb[lambda_2]["AE"] = angular_error_tpb
        results_tpb[lambda_2]["L2_error"] = l2_loss_tpb
        results_tpb[lambda_2]["Normalized_L2_error"] = norm_l2_loss_tpb
        results_tpb[lambda_2]["hell_loss"] = hellinger_loss_tpb
        results_tpb[lambda_2]["l2_corr"] = l2_correlation_tpb
        results_tpb[lambda_2]["Approx_ISE"] = approx_ise_crit[-1]

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
                                extrincis(quad_tensor[:,1])), dim=1)
quad_tensor_euc = quad_tensor_euc.to(device)

domain = "sphere"
measure = "uniformsphere"
activation = "exp" 
encoding = "ident"

alpha = 1.0
hidden_width = 181
output_width = 181
marg_dim = 2

snef_model = snef_prod.SquaredNNProd(D, domain, measure, activation, encoding, d=marg_dim, m=output_width, n=hidden_width)
snef_model = snef_model.to(device)

## optimization params 
lr = 5e-5; num_epochs = 10000; 
optim = torch.optim.Adam(snef_model.parameters(), lr=lr)

## random split 
train_prop = 0.9; batch_frac = 1
n_train = int(train_prop * n)
batch_size = n//batch_frac
indices = torch.randperm(n) 

points_tensor_euc = torch.cat((extrincis(points_tensor[:,0]), 
                                extrincis(points_tensor[:,1])), dim=1)
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
            density_hat_tensor = ((norm_pdf)**2) * torch.exp(log_fhat - log_norm_integral).cpu().detach()
            l2_loss_norm = torch.mean((true_density_evals - density_hat_tensor)**2).item()/torch.mean((true_density_evals**2)).item()
            hellinger_loss = torch.mean((torch.sqrt(true_density_evals) - torch.sqrt(density_hat_tensor))**2).item()
            l2_correlation = pearsonr(true_density_evals.cpu().detach().numpy().flatten(), density_hat_tensor.numpy().flatten())[0]
            angular_error = np.arccos(pearsonr(np.sqrt(true_density_evals.cpu().detach().numpy().flatten()), np.sqrt(density_hat_tensor.numpy().flatten()))[0])
            print(epoch, "nISE", l2_loss_norm, "Angular Error", angular_error, "Integrand", density_hat_tensor.mean().item())
            writer.add_scalar('Metrics/max_value', density_hat_tensor.max().item(), epoch)
            writer.add_scalar('Loss/L2_loss', l2_loss_norm, epoch)
            writer.add_scalar('Loss/Hellinger_Loss', hellinger_loss, epoch)
            writer.add_scalar('Loss/L2_corr', l2_correlation, epoch)
            writer.add_scalar('Loss/AE', angular_error, epoch)

t2_psnf = time.time() - t1_psnf

log_fhat = snef_model(quad_tensor_euc, log_scale=True)
log_norm_integral = snef_model.integrate(log_scale=True)
density_hat_tensor = ((norm_pdf)**2) * torch.exp(log_fhat - log_norm_integral).cpu().detach()

density_matrix_hat = density_hat_tensor.reshape(Qgrid, Qgrid).cpu().detach().numpy()
true_density_matrix = true_density_evals.reshape(Qgrid, Qgrid).cpu().detach().numpy()

angular_error_snef = np.arccos(pearsonr(np.sqrt(true_density_matrix.flatten()), np.sqrt(density_matrix_hat.flatten()))[0])
l2_loss_snef = (np.linalg.norm(true_density_matrix - density_matrix_hat, ord="fro")**2)*(1./(Qgrid**2))
norm_l2_loss_snef = l2_loss_snef/((np.linalg.norm(true_density_matrix, ord="fro")**2)*(1./(Qgrid**2)))


results_psnf["time"] = t2_psnf
results_psnf["AE"] = angular_error_snef
results_psnf["L2_error"] = l2_loss_snef
results_psnf["Normalized_L2_error"] = norm_l2_loss_snef

with open(os.path.join(out_dir, "results_snef_%s.pkl"%mc_exp), "wb") as pklfile:
    pickle.dump(results_psnf, pklfile)

