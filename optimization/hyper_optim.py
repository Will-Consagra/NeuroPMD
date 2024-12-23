import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np 
from itertools import product

from sklearn.model_selection import KFold

import os 
import sys 

PATH2NCP = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.append(os.path.join(PATH2NCP, "neuroPMD"))
from data_object import PointPattern

def _evaluate(device, hyper_params, field_model, optimizer, algorithm, dataloader_train, points_tensor_test):
	## estimate parameters 
	algorithm(device, 
			dataloader_train,
			field_model,
			optimizer,
			hyper_params)
	# Calculate MISE and output
	## approximate \int(f - \hat{f})^2 using validation set. 
	## 0) compute normalization integral on quadrature points
	quad_tensor_valid = hyper_params["quad_tensor"]
	Ntest = points_tensor_test.size(0)
	Q = quad_tensor_valid.size(0)
	dA = torch.tensor(1/Q)
	model_quad_eval_i = field_model(quad_tensor_valid.to(device))["model_out"]
	log_norm_integral = torch.log(dA*(torch.exp(model_quad_eval_i)).sum())
	## 1) compute \hat{f} on validation sets
	log_fhat = field_model(points_tensor_test.to(device))["model_out"]
	fmodel_valid_i = torch.exp(log_fhat - log_norm_integral)
	## 2) compute \int \hat{f}^2 using quadrature points
	fmodel_l2_norm_quadrature_i = torch.square(torch.exp(model_quad_eval_i - log_norm_integral))*dA
	## 3) compute [\int (f-\hat{f})^2] - \int f^2 = \int \hat{f}^2 - 2\int f\hat{f}
	approx_l2_error_i = fmodel_l2_norm_quadrature_i.sum() - (2./Ntest)*fmodel_valid_i.sum() 
	return {"L2": (approx_l2_error_i.cpu().detach().numpy(), 0.0)}

def validation_set_selection(points_tensor, field_model_, optimizer_, algorithm, hyper_params, parameter_map, train_prop=0.9, batch_frac=1, num_iter_track=200):
	param_names = [param["name"] for param in parameter_map]
	param_values = [param["vals"] for param in parameter_map]
	param_combinations = list(product(*param_values))
	avg_val_losses = []
	device = hyper_params["quad_tensor"].device ## quad tensor already on GPU 
	num_epochs = hyper_params["num_epochs"]
	D = hyper_params["D"]	
	## get small validation set 
	n = points_tensor.size(0) 
	n_train = int(train_prop * n)
	batch_size = n_train//batch_frac
	indices = torch.randperm(n) 
	points_tensor_train = points_tensor[indices[:n_train],:]
	points_tensor_test = points_tensor[indices[n_train:],:]
	O_train = PointPattern(points_tensor_train)
	dataloader_train = DataLoader(O_train, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=0)
	## store results 
	results = []
	for param_comb in param_combinations:
		params = dict(zip(param_names, param_comb))
		full_params = {**hyper_params, **params}
		## get field model params 
		fmodel_params = {}
		if "rank" in full_params:
			fmodel_params["rank"] = full_params["rank"]
		if "width" in full_params:
			fmodel_params["width"] = full_params["width"]  
		if "depth" in full_params:
			fmodel_params["depth"] = full_params["depth"]  
		if "out_channels" in full_params:
			fmodel_params["out_channels"] = full_params["out_channels"]  
		if "w0" in full_params:
			fmodel_params["w0"] = full_params["w0"] 
		if "max_freq" in full_params:
			fmodel_params["max_freq"] = full_params["max_freq"]  
		## optimzation parameters 
		num_epochs = full_params["num_epochs"] 
		lr = full_params["lr"]
		## intiate field model and evaluate criteria 
		field_model = field_model_(**fmodel_params).to(device)
		optim = optimizer_(params=field_model.parameters(), lr = lr)
		## run optimization 
		full_params["points_tensor_test"] = points_tensor_test
		approx_ise_crit = algorithm(device, dataloader_train, field_model, optim, full_params, num_iter_track=num_iter_track)
		results.append({"train_modeled":field_model, "criteria":approx_ise_crit, "parameters":params})
		print("Completed trail for ")
		print(params, approx_ise_crit[-1])
	return results

def CV_optim(nfolds, points_tensor, field_model_, optimizer_, algorithm, hyper_params, parameter_map, batch_frac=2):
	kf = KFold(n_splits=nfolds)
	param_names = [param["name"] for param in parameter_map]
	param_values = [param["vals"] for param in parameter_map]
	param_combinations = list(product(*param_values))
	avg_val_losses = []
	device = hyper_params["quad_tensor"].device ## quad tensor already on GPU 
	num_epochs = hyper_params["num_epochs"]
	D = hyper_params["D"]
	for param_comb in param_combinations:
		params = dict(zip(param_names, param_comb))
		full_params = {**hyper_params, **params}
		## get field model params 
		fmodel_params = {}
		if "rank" in full_params:
			fmodel_params["rank"] = full_params["rank"]
		if "width" in full_params:
			fmodel_params["width"] = full_params["width"]  
		if "depth" in full_params:
			fmodel_params["depth"] = full_params["depth"]  
		if "out_channels" in full_params:
			fmodel_params["out_channels"] = full_params["out_channels"]  
		if "w0" in full_params:
			fmodel_params["w0"] = full_params["w0"] 
		if "max_freq" in full_params:
			fmodel_params["max_freq"] = full_params["max_freq"]  
		## optimzation parameters 
		num_epochs = full_params["num_epochs"] 
		lr = full_params["lr"]
		## fit the folds 
		val_losses = []
		for train_index, val_index in kf.split(points_tensor):
			## split data
			points_tensor_train = points_tensor[train_index,:]
			points_tensor_test = points_tensor[val_index,:]
			batch_size = points_tensor.size(0)//batch_frac
			O_train = PointPattern(points_tensor_train)
			dataloader_train = DataLoader(O_train, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=0)
			## instantiate model and optimzier
			field_model = field_model_(**fmodel_params).to(device)
			optim = optimizer_(params=field_model.parameters(), lr = lr)
			val_loss = _evaluate(device, full_params, field_model, optim, algorithm, dataloader_train, points_tensor_test)
			val_losses.append(val_loss["L2"])
		avg_val_losses.append(np.mean(val_losses))
		print("Completed trail for ")
		print(param_comb)
	min_ix = np.argmin(avg_val_losses)
	param_comb_optim = param_combinations[min_ix]
	best_parameters = dict(zip(param_names, param_comb_optim))
	return best_parameters, {"param_list":param_combinations, "param_nams":param_names, "avg_val_losses":avg_val_losses}

