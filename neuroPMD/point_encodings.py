import numpy as np
import torch
from torch import nn
from scipy.special import sph_harm
import random
import itertools 
from utils import cart2sphere
from sph_harm_utils import get_spherical_harmonics_element, clear_spherical_harmonics_cache

class RandomTphFourier(nn.Module):
	#Random subset of non-separable Harmonics on hypertorus 
	def __init__(self, rank, D=2, max_freq=20):
		super().__init__()
		self.max_freq = max_freq
		self.freq_set = list(itertools.product(range(1, self.max_freq+1), repeat=D))
		self.rank = rank
		if self.rank < (self.max_freq**D):
			self.rand_ix = random.sample(self.freq_set, self.rank)
			self.register_buffer("frequency_matrix", torch.tensor([list(tup) for tup in self.rand_ix]).float())
		else: ## default to full tensor product basis 
			self.register_buffer("frequency_matrix", torch.tensor(self.freq_set).float())
	def forward(self, coordinates):
		inner_products = torch.matmul(coordinates, self.frequency_matrix.T) 
		return torch.cat((torch.cos(inner_products),
					torch.sin(inner_products)), dim=-1)

class FourierBasis(nn.Module):
	#Marginal Fourier basis 
	def __init__(self, n_basis=3):
		super().__init__()
		self.period = 2*torch.pi 
		# If the number of basis is even, add 1
		self.n_basis = n_basis + (1 - n_basis % 2)
	def forward(self, eval_points: torch.Tensor):
		"""
		Evaluate the Fourier basis functions at given points.

		Parameters:
		- eval_points: Tensor of points to evaluate, of shape (n_points,)

		Returns:
		- Tensor of evaluated basis functions, of shape (n_basis, n_points)
		"""
		# Ensure eval_points are on the correct dimension
		# Define sine and cosine functions
		omega = 2 * np.pi / self.period
		normalization_denominator = np.sqrt(self.period / 2)
		seq = 1 + torch.arange((self.n_basis - 1) // 2, device=eval_points.device, dtype=eval_points.dtype)
		phase_coefs_tensor = omega * torch.vstack((seq, seq)).T
		# Compute the arguments for sine and cosine functions
		res = torch.einsum('ij,k->ijk', phase_coefs_tensor, eval_points)
		# Apply sine and cosine to each phase component
		res_sin = torch.sin(res[:, 0, :])
		res_cos = torch.cos(res[:, 1, :])
		result_full = torch.cat([res_sin, res_cos], dim=0) / normalization_denominator  # Shape: (n_points, n_basis-1)
		# Add the constant basis function
		constant_basis = (eval_points * 0 + 1).unsqueeze(0) / (torch.sqrt(torch.tensor(2.0, dtype=eval_points.dtype, device=eval_points.device)) * normalization_denominator)
		# Concatenate and return
		return torch.cat((constant_basis, result_full), dim=0).T

class RandomTphFourierPyT(nn.Module):
	#Random subset of non-separable Harmonics on hypertorus 
    def __init__(self, rank, D=2, max_freq=20):
        super().__init__()
        basis_list = [FourierBasis(max_freq) for d in range(D)]
        self.basis_list = nn.ModuleList(basis_list)
        self.dim_domain = len(self.basis_list)
        self.n_basis_per_dim = [b.n_basis for b in self.basis_list]
        self.n_basis_total = torch.prod(torch.tensor(self.n_basis_per_dim))
        self.rank = rank
        # Randomly sample K indices from the full tensor product basis function space
        self.sampled_indices = torch.randperm(self.n_basis_total)[:self.rank]
        # Precompute the tensor indices corresponding to the sampled basis functions
        grids = torch.meshgrid(*[torch.arange(n) for n in self.n_basis_per_dim], indexing='ij')
        indices = torch.stack(grids, dim=-1).reshape(-1, self.dim_domain)  # Shape: (n_basis_total, dim_domain)
        self.sampled_tensor_indices = indices[self.sampled_indices]
    def forward(self, eval_points: torch.Tensor):
        n_points, dim = eval_points.shape
        if dim != self.dim_domain:
            raise ValueError(f"Expected points of dimension {self.dim_domain}, but got {dim}.")
        # Evaluate Fourier basis for each dimension
        basis_evaluations = [b(eval_points[:, i]).T for i, b in enumerate(self.basis_list)]
        # Initialize result tensor on the same device
        result = torch.ones((self.rank, n_points), device=eval_points.device)
        # Compute the full tensor product by multiplying the basis evaluations for sampled indices
        for i in range(self.dim_domain):
            result = result * basis_evaluations[i][self.sampled_tensor_indices[:, i]]
        return result.T

class FullTensorProductFourierBasisPyT(nn.Module):
	#Full separable tensor product Harmonics on hypertorus 
	def __init__(self, basis_list):
		"""
		Initialize the TensorProductFourierBasis object.
		Parameters:
		- basis_list: List of FourierBasis instances for each dimension.
		"""
		super().__init__()
		self.basis_list = nn.ModuleList(basis_list)
		self.dim_domain = len(self.basis_list)
		self.n_basis_per_dim = [b.n_basis for b in self.basis_list]
		self.n_basis_total = torch.prod(torch.tensor([b.n_basis for b in self.basis_list]))
	def forward(self, eval_points: torch.Tensor):
		"""
		Evaluate the tensor product of Fourier basis functions at given points.

		Parameters:
		- eval_points: Tensor of points to evaluate, of shape (n_points, dim_domain)

		Returns:
		- Tensor of evaluated tensor product basis functions, of shape (n_basis_total, n_points)
		"""
		n_points, dim = eval_points.shape
		if dim != self.dim_domain:
			raise ValueError(f"Expected points of dimension {self.dim_domain}, but got {dim}.")
		# Evaluate Fourier basis for each dimension
		basis_evaluations = [b(eval_points[:, i]).T for i, b in enumerate(self.basis_list)]
		# Compute the full tensor product using broadcasting and outer product
		grids = torch.meshgrid(*[torch.arange(n) for n in self.n_basis_per_dim], indexing='ij')
		indices = torch.stack(grids, dim=-1).reshape(-1, self.dim_domain)  # Shape: (n_basis_total, dim_domain)
		# Initialize result tensor on the same device
		result = torch.ones((self.n_basis_total, n_points), device=eval_points.device)
		# Compute the full tensor product by multiplying the basis evaluations
		for i in range(self.dim_domain):
			result = result * basis_evaluations[i][indices[:, i]]
		return result.T

class RTPSphericalHarmonicsPyT(nn.Module):
	#Random subset of tensor product Harmonics 
	def __init__(self, rank, max_degree=20):
		super().__init__()
		self.max_degree = max_degree
		self.K = (max_degree+1)**2
		self.l = torch.arange(0,max_degree+1)
		self.m = torch.arange(-max_degree,max_degree+1)
		self.rank = rank
		## randomly sample sample degrees and orders
		self.ixmap = {}
		for l in range(self.max_degree+1):
			for m_l in range(-l,l+1):
				ix = self._get_flax_index(l, m_l)
				self.ixmap[ix] = (l, m_l)
		ix_pairs = [(i1, i2) for i1 in self.ixmap.keys() for i2 in self.ixmap.keys()]
		self.rand_ix = random.sample(ix_pairs, self.rank)
	def _get_flax_index(self, l, m_l):
		return l*(l + 1) + m_l
	def _spharm(self, inc, azim):
		Psi = torch.zeros(tuple(list(inc.shape) + [self.K]), device=inc.device)
		for l in range(self.max_degree+1):
			for m_l in range(-l,l+1):
				Y_ml = get_spherical_harmonics_element(l, m_l, inc, azim)
				clear_spherical_harmonics_cache()
				idx = self._get_flax_index(l, m_l)
				Psi[...,idx] = Y_ml
		return Psi
	def forward(self, coordinates):
		## still inefficient but faster than previous forward pass algorithm
		device = coordinates.device
		coordinates = coordinates
		Omega_1 = coordinates[...,:3]; Omega_2 = coordinates[...,3:]; 
		Omega_1_sph = cart2sphere(Omega_1); Omega_2_sph = cart2sphere(Omega_2)
		azimuthal_1 = Omega_1_sph[...,0]; inclination_1 = Omega_1_sph[...,1];
		azimuthal_2 = Omega_2_sph[...,0]; inclination_2 = Omega_2_sph[...,1];
		Psi_1 = self._spharm(inclination_1, azimuthal_1)
		Psi_2 = self._spharm(inclination_2, azimuthal_2)
		Psi_prod = torch.zeros(*coordinates.shape[:-1], self.rank, device=device)
		for k, (i, j) in enumerate(self.rand_ix):
			Psi_prod[..., k] = Psi_1[..., i] * Psi_2[..., j]
		return Psi_prod

class SphericalHarmonics(nn.Module):
	#Spherical harmonics 
	def __init__(self, max_degree=20):
		super().__init__()
		self.max_degree = max_degree
		self.K = (max_degree+1)**2
		self.l = torch.arange(0,max_degree+1)
		self.m = torch.arange(-max_degree,max_degree+1)
	def _get_real_shm(self, Yml_cmplx, m, l):
		if m > 0:
			Y_ml = np.sqrt(2) * Yml_cmplx.real
		elif m == 0:
			Y_ml = Yml_cmplx
		elif m < 0:
			Y_ml = np.sqrt(2) * Yml_cmplx.imag
		return Y_ml
	def _get_flax_index(self, l, m_l):
		return l*(l + 1) + m_l
	def _spharm(self, inc, azim):
		Psi = torch.zeros(tuple(list(inc.shape) + [self.K]))
		for l in range(self.max_degree+1):
			for m_l in range(-l,l+1):
				Yml_cmplx = sph_harm(abs(m_l), l, azim.cpu(), inc.cpu())
				Y_ml = self._get_real_shm(Yml_cmplx, m_l, l)
				idx = self._get_flax_index(l, m_l)
				Psi[...,idx] = Y_ml
		return Psi
	def forward(self, coordinates):
		## still inefficient but faster than previous forward pass algorithm
		device = coordinates.device
		Omega_sph = cart2sphere(coordinates)
		azimuthal = Omega_sph[...,0]; inclination = Omega_sph[...,1];
		Psi = self._spharm(inclination, azimuthal).to(device)
		return Psi 

class TPHarmonics(nn.Module):
	""" Tensor product real spherical harmonics.

	Args:
		max_degree (int): Maximum degree of marginal harmonic basis 
	"""
	def __init__(self, max_degree=8):
		super().__init__()
		self.max_degree = max_degree
		self.K = (max_degree+1)**2
		self.l = torch.arange(0,max_degree+1)
		self.m = torch.arange(-max_degree,max_degree+1)
		self.rank = self.K**2
		
	def _get_real_shm(self, Yml_cmplx, m, l):
		if m > 0:
			Y_ml = np.sqrt(2) * Yml_cmplx.real
		elif m == 0:
			Y_ml = Yml_cmplx
		elif m < 0:
			Y_ml = np.sqrt(2) * Yml_cmplx.imag
		return Y_ml

	def _get_flax_index(self, l, m_l):
		return l*(l + 1) + m_l

	def _spharm(self, inc, azim):
		Psi = torch.zeros(tuple(list(inc.shape) + [self.K]))
		for l in range(self.max_degree+1):
			for m_l in range(-l,l+1):
				Yml_cmplx = sph_harm(abs(m_l), l, azim.cpu(), inc.cpu())
				Y_ml = self._get_real_shm(Yml_cmplx, m_l, l)
				idx = self._get_flax_index(l, m_l)
				Psi[...,idx] = Y_ml
		return Psi

	def forward(self, coordinates):
		"""Spherical harmonic basis expansion
		Args:
			coordinates (torch.Tensor): Shape (..., num_points, 6)
		"""
		device = coordinates.device
		coordinates = coordinates

		Omega_1 = coordinates[...,:3]; Omega_2 = coordinates[...,3:]; 
		Omega_1_sph = cart2sphere(Omega_1); Omega_2_sph = cart2sphere(Omega_2)
		azimuthal_1 = Omega_1_sph[...,0]; inclination_1 = Omega_1_sph[...,1];
		azimuthal_2 = Omega_2_sph[...,0]; inclination_2 = Omega_2_sph[...,1];

		Psi_1 = self._spharm(inclination_1, azimuthal_1).to(device)
		Psi_2 = self._spharm(inclination_2, azimuthal_2).to(device)
		return torch.einsum('...i,...j->...ij', Psi_1, Psi_2).view(*Psi_1.shape[:-1], -1)

