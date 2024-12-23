import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math

from typing import Tuple, Callable, Dict

from point_encodings import TPHarmonics, RandomTphFourier, FourierBasis, FullTensorProductFourierBasisPyT, RandomTphFourierPyT, RTPSphericalHarmonicsPyT

class SineLayer(nn.Module):
	def __init__(self, in_features, out_features, bias=True,
				 is_first=False, omega_0=30):
		super().__init__()
		self.omega_0 = omega_0
		self.is_first = is_first
		self.in_features = in_features
		self.linear = nn.Linear(in_features, out_features, bias=bias)
		self.init_weights()
	def init_weights(self):
		with torch.no_grad():
			if self.is_first:
				self.linear.weight.uniform_(-1 / self.in_features, 
											 1 / self.in_features)      
			else:
				self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
											 np.sqrt(6 / self.in_features) / self.omega_0)	
	def forward(self, input):
		return torch.sin(self.omega_0 * self.linear(input))  

class PPLatentToroidalSiren(nn.Module):
	#NeuroPMD for T^D
	def __init__(self, D: int = 2, max_freq: int = 20, rank: int = 50, width: int = 256, depth: int = 5, out_channels: int = 1,
					w0: float = 1., outermost_linear: bool = True, sep_enc: bool = True):
		super().__init__()
		self.D = D 
		self.width = width
		self.depth = depth
		self.out_channels = out_channels
		self.w0 = w0
		## define base network structure
		if sep_enc:
			self.tph_layer = RandomTphFourierPyT(rank, D=D, max_freq=max_freq)
			initial_layer =  SineLayer(self.tph_layer.rank, self.width, is_first=True, omega_0=w0)
		else:
			self.tph_layer = RandomTphFourier(rank, D=D, max_freq=max_freq)
			initial_layer =  SineLayer(2*self.tph_layer.rank, self.width, is_first=True, omega_0=w0)
		if outermost_linear:
			final_layer = nn.Linear(self.width, self.out_channels, bias=True)
			with torch.no_grad(): ##same initialization as the sine layers 
				final_layer.weight.uniform_(-np.sqrt(6 / self.width) / self.w0,
											 np.sqrt(6 / self.width) / self.w0)
		else:
			final_layer = SineLayer(self.width, self.out_channels, bias=True, is_first=False, omega_0=self.w0)
		self.base_network = nn.ModuleList([initial_layer] + [SineLayer(self.width, self.width, bias=True, is_first=False, omega_0=self.w0) for i in range(1, self.depth - 1)] + [final_layer])
	def forward(self, x):
		fevals = self.tph_layer(x)
		for index, layer in enumerate(self.base_network): 
			fevals = layer(fevals)
		return {"model_in": x, "model_out": fevals}
	def forward_wgrad(self, x):
		x = x.clone().detach().requires_grad_(True)
		fevals = self.tph_layer(x)
		for index, layer in enumerate(self.base_network): 
			fevals = layer(fevals)
		return {"model_in": x, "model_out": fevals}

class PPLatentSphericalSiren(nn.Module):
	# NeuroPMD for S2 x S2 
	def __init__(self, max_degree: int = 8, rank: int = 0, width: int = 256, depth: int = 5, out_channels: int = 1,
					w0: float = 1., outermost_linear: bool = True):
		super().__init__()
		self.width = width
		self.depth = depth
		self.out_channels = out_channels
		self.w0 = w0
		## create encoder 
		if not rank:
			self.sph_layer =  TPHarmonics(max_degree)
		else:
			self.sph_layer =  RTPSphericalHarmonicsPyT(rank, max_degree=max_degree)
		# build MLP layers
		layers = []
		for i in range(depth-1):
			if not i:
				layers.append(SineLayer(self.sph_layer.rank, self.width, bias=True, is_first=True, omega_0=self.w0)) 
			else:
				layers.append(SineLayer(self.width, self.width, bias=True, is_first=False, omega_0=self.w0))
		final_layer = nn.Linear(self.width, self.out_channels, bias=True)
		with torch.no_grad(): ##same initialization as the sine layers 
			final_layer.weight.uniform_(-np.sqrt(6 / self.width) / self.w0,
										 np.sqrt(6 / self.width) / self.w0)
		layers.append(final_layer)
		self.base_network = nn.ModuleList(layers)
	def forward(self, x):
		fevals = self.sph_layer(x)
		for index, layer in enumerate(self.base_network): 
			fevals = layer(fevals)
		return {"model_in": x, "model_out": fevals}
	def forward_wgrad(self, x):
		x = x.clone().detach().requires_grad_(True)
		fevals = self.sph_layer(x)
		for index, layer in enumerate(self.base_network): 
			fevals = layer(fevals)
		return {"model_in": x, "model_out": fevals}

class TPFBLinearModel_NS(nn.Module):
	def __init__(self, max_freq, D):
		super().__init__()
		self.rank = max_freq**D
		self.tph_layer = RandomTphFourier(self.rank, D=D, max_freq=max_freq)
		self.linear = nn.Linear(2*self.tph_layer.rank, 1)
	def forward(self, x):
		fevals = self.tph_layer(x)
		fevals = self.linear(fevals)
		return {"model_in": x, "model_out": fevals}
	def forward_wgrad(self, x):
		x = x.clone().detach().requires_grad_(True)
		fevals = self.tph_layer(x)
		fevals = self.linear(fevals)
		return {"model_in": x, "model_out": fevals}

class TPFBLinearModel(nn.Module):
	def __init__(self, max_freq, D):
		super().__init__()
		marg_basis_list = [FourierBasis(max_freq) for d in range(D)]
		self.tph_layer = FullTensorProductFourierBasisPyT(marg_basis_list)
		self.linear = nn.Linear(self.tph_layer.n_basis_total, 1)
		self.rank = self.tph_layer.n_basis_total
	def forward(self, x):
		fevals = self.tph_layer(x)
		fevals = self.linear(fevals)
		return {"model_in": x, "model_out": fevals}
	def forward_wgrad(self, x):
		x = x.clone().detach().requires_grad_(True)
		fevals = self.tph_layer(x)
		fevals = self.linear(fevals)
		return {"model_in": x, "model_out": fevals}

class SphHarmTPLinModel(nn.Module):
	def __init__(self, max_degree=8):
		super().__init__()
		self.sph_layer =  TPHarmonics(max_degree)
		self.linear = nn.Linear(self.sph_layer.rank, 1)
	def forward(self, x):
		fevals = self.sph_layer(x)
		fevals = self.linear(fevals)
		return {"model_in": x, "model_out": fevals}
	def forward_wgrad(self, x):
		x = x.clone().detach().requires_grad_(True)
		fevals = self.sph_layer(x)
		fevals = self.linear(fevals)
		return {"model_in": x, "model_out": fevals}

class MLP(nn.Module):
	def __init__(self, in_features, hidden_features, hidden_layers, out_features, activation=nn.ReLU(), dropout=0.0):
		super().__init__()
		layers = []
		# Add the first linear layer to handle input channels
		layers.append(nn.Linear(in_features, hidden_features))
		layers.append(activation)
		if dropout > 0.0:
			layers.append(nn.Dropout(dropout))
		for i in range(1, hidden_layers):
			layers.append(nn.Linear(hidden_features, hidden_features))
			layers.append(activation)
			if dropout > 0.0:
				layers.append(nn.Dropout(dropout))
		layers.append(nn.Linear(hidden_features, out_features))
		self.mlp = nn.Sequential(*layers)
	def forward(self, x):
		return self.mlp(x)
