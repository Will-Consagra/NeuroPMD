import torch 
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import itertools 

from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter

from torch.autograd import grad
from torch.optim.lr_scheduler import CyclicLR

## Diff Operators 
def gradient(y, x, grad_outputs=None):
	if grad_outputs is None:
		grad_outputs = torch.ones_like(y)
	grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]
	return grad

def divergence(y, x):
	div = 0.
	for i in range(y.shape[-1]):
		div += grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True, retain_graph=True)[0][..., i:i+1]
	return div

def laplace(y, x):
	grad = gradient(y, x)
	return divergence(grad, x)

def grad_discrete(model, batch_points, eps):
	device = batch_points.device
	n, d = batch_points.shape
	gradients = torch.zeros_like(batch_points)
	perturb = torch.zeros_like(batch_points)
	for i in range(d):
		# Create perturbation vectors
		perturb[:, i] = eps
		# Compute function values at perturbed batch_points
		f_plus = model(batch_points + perturb)["model_out"]
		f_minus = model(batch_points - perturb)["model_out"]
		# Compute finite difference approximation of the gradient
		gradients[:, i] = ((f_plus - f_minus) / (2 * eps)).flatten()
		## clean up 
		perturb[:, i] = 0.
	return gradients

def laplace_discrete(model, batch_points, eps):
	n, d = batch_points.shape
	hess_diag = torch.zeros_like(batch_points)
	f_center = model(batch_points)["model_out"]
	perturb = torch.zeros_like(batch_points)
	for i in range(d):
		# Create perturbation vectors
		perturb[:, i] = eps
		# Compute function values at perturbed points
		f_plus = model((batch_points + perturb))["model_out"]
		f_minus = model((batch_points - perturb))["model_out"]
		# Compute finite difference approximation of the second derivative
		hess_diag[:,i] = ((f_plus - 2 * f_center + f_minus) / (eps ** 2)).flatten()
		## clean up 
		perturb[:, i] = 0.
	return hess_diag.sum(dim=-1)

def MAP_f(device, dataloader, func_model, optim, hyper_params, DiffOpCalcDisc=False, eps=1e-3, num_iter_track=200):
	# Run SGA based optimization 
	## get hyper-parameters 
	approx_ise_crit = []
	num_epochs =  hyper_params["num_epochs"]
	MC_Q = hyper_params["MC_Q"]
	MQ_Q_DO = hyper_params["MQ_Q_DO"]
	D = hyper_params["D"]
	lambda_1 = hyper_params.get("lambda_1")
	lambda_2 = hyper_params.get("lambda_2")
	mc_sampler = hyper_params["mc_sampler"]
	outname = hyper_params["outname"]
	Vol = hyper_params.get("Volume", 1)
	if hyper_params["cycle_lr"]:
		scheduler = CyclicLR(optim, base_lr=hyper_params["base_lr"], max_lr=hyper_params["max_lr"], step_size_up=num_epochs//hyper_params["step_size_up_frac"], mode=hyper_params["cycle_mode"])
	if outname:
		writer = SummaryWriter(outname)  
	if "true_density_evals" in hyper_params:
		from scipy.stats import pearsonr
		true_density_evals = hyper_params["true_density_evals"]
		quad_tensor = hyper_params["quad_tensor"].to(device)
		dA = Vol/quad_tensor.shape[0]
		TRACK_GT = True
	else:
		TRACK_GT = False
	if "points_tensor_test" in hyper_params:
		points_tensor_test = hyper_params["points_tensor_test"]
		quad_tensor = hyper_params["quad_tensor"].to(device)
		dA = Vol/quad_tensor.shape[0]
		TRACK_ISE = True
	else:
		TRACK_ISE = False
	with tqdm(total=len(dataloader)*num_epochs) as pbar:
		for epoch in range(num_epochs+1):
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
				#	grad_d = grad_discrete(func_model, mc_diff_op_points_tensor, eps)
				#	TV_prior = (Vol/MQ_Q_DO)*((torch.abs(grad_d)).sum())
				#	lap_d = laplace_discrete(func_model, mc_diff_op_points_tensor, eps)
				#	Rough_prior = (Vol/MQ_Q_DO)*(lap_d**2).sum()
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
				if hyper_params["cycle_lr"]:
					scheduler.step()
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
				log_norm_integral = torch.log(dA*(torch.exp(func_model(quad_tensor)["model_out"])).sum())
				density_hat_tensor = torch.exp(log_fhat - log_norm_integral).cpu().detach()
				l2_loss = torch.mean((true_density_evals - density_hat_tensor)**2).item()
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
				log_norm_integral = torch.log(dA*(torch.exp(model_quad_eval_i)).sum())
				## 1) compute \hat{f} on validation sets
				log_fhat = func_model(points_tensor_test.to(device))["model_out"]
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
	return approx_ise_crit

