import torch
from torch import nn
import numpy as np 
from scipy.stats import vonmises, vonmises_fisher
from sklearn.model_selection import KFold

class Toroidal_KDE(nn.Module):
	def __init__(self, observed_data, norm_term=1):
		super().__init__()
		self.observed_data = observed_data
		self.norm_term = norm_term 
		self.D = observed_data.shape[1]
		self.n = observed_data.shape[0]
	def forward(self, coordinates, kappa):
		## coordinates: nbatch x D -> observed data, each d is an angle in [-pi, pi)
		## kappa: float: common bandwidth parameter 
		K_TD = torch.zeros(coordinates.shape[0], self.n, self.D)
		for i in range(coordinates.shape[0]):
			angle_i = coordinates[i,...]
			for d in range(self.D):
				K_TD[i,:,d] = torch.from_numpy(vonmises.pdf(angle_i[d], kappa[d], self.observed_data[:,d])).float()
		return (self.norm_term**self.D)*torch.mean(torch.prod(K_TD, axis=2),axis=1)

class S2xS2_KDE(nn.Module):
	def __init__(self, observed_data, norm_term=1):
		super().__init__()
		self.observed_data = observed_data
		self.norm_term = norm_term 
		self.D = 2
		self.n = observed_data.shape[0]
	def forward(self, coordinates, kappa):
		## coordinates: nbatch x 6 -> observed data
		## kappa: float: common bandwidth parameter 
		K_SD = torch.zeros(coordinates.shape[0], self.n, self.D)
		for i in range(coordinates.shape[0]):
			coord_i = coordinates[i,...]
			for d in range(self.D):
				K_SD[i,:,d] = torch.from_numpy(vonmises_fisher.pdf(self.observed_data[:,d*3:(d+1)*3], mu=coord_i[d*3:(d+1)*3], kappa=kappa[d].item())).float()
		return (self.norm_term**self.D)*torch.mean(torch.prod(K_SD, axis=2),axis=1)

def kfold_BW_selector(kappa_values, observed_data, quad_tensor, quad_weights, norm_term, domain="Torus", nsplits=5):
	D = observed_data.shape[1]
	kf = KFold(n_splits=nsplits, shuffle=True)
	avg_ise_scores = []
	for kappa in kappa_values:
		scores = []
		kappa_tensor = torch.tensor([kappa]*D)
		for train_index, test_index in kf.split(observed_data):
			train_data, test_data = observed_data[train_index,:], observed_data[test_index,:]
			ntest = test_data.shape[0]
			if domain=="Torus":
				kde_i = Toroidal_KDE(train_data, norm_term=norm_term)
			elif domain=="S2xS2":
				kde_i = S2xS2_KDE(train_data, norm_term=norm_term)
			## 1) Calculate l2 norm 
			f_hat_quad = kde_i(quad_tensor, kappa_tensor)
			f_hat_l2 = (torch.square(f_hat_quad) * quad_weights).sum()
			## 2) compute <\hat{f}, f>
			f_hat_test = kde_i(test_data, kappa_tensor)
			## 3) compute ISE score 
			score_i = f_hat_l2 - ((2./ntest) * f_hat_test.sum())
			scores.append(score_i.item())
		average_score = np.mean(scores)
		avg_ise_scores.append(average_score)
		print("Finished kappa = ", kappa)
	return avg_ise_scores


