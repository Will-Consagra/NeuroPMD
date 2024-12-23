import numpy as np 
import torch

def cart2sphere(x):
	"""
	theta: azimuthal AKA longitudinal
	phi: inclination AKA colatitude AKA polar
	"""
	#r = np.sqrt(x[:,0]**2 + x[:,1]**2 + x[:,2]**2)
	#theta = np.arctan2(x[:,1], x[:,0])
	#phi = np.arccos(x[:,2]/r)
	#return np.column_stack([theta, phi])
	r = torch.sqrt(x[...,0]**2 + x[...,1]**2 + x[...,2]**2)
	theta = torch.arctan2(x[...,1], x[...,0]) ##azimuth
	phi = torch.arccos(x[...,2]/r) ##inclination
	return torch.stack((theta, phi), dim=len(theta.shape))

def sphere2cart(x):
	"""
	theta: azimuthal AKA longitudinal
	phi: inclination AKA colatitude AKA polar
	"""
	#theta = x[:,0]
	#phi = x[:,1]
	#xx = np.sin(phi)*np.cos(theta)
	#yy = np.sin(phi)*np.sin(theta)
	#zz = np.cos(phi)
	#return np.column_stack([xx, yy, zz]) 
	theta = x[...,0]
	phi = x[...,1]
	xx = torch.sin(phi)*torch.cos(theta)
	yy = torch.sin(phi)*torch.sin(theta)
	zz = torch.cos(phi)
	return torch.stack((xx, yy, zz), dim=len(theta.shape))

def S2_get_tangent_basis(coords_sph):
	azimuthal = coords_sph[...,0:1]; inclination = coords_sph[...,1:];
	u1 = torch.cat((torch.cos(inclination)*torch.cos(azimuthal), 
					torch.cos(inclination)*torch.sin(azimuthal),
					-torch.sin(inclination)) ,dim=-1)
	u2 = torch.cat((-torch.sin(inclination)*torch.sin(azimuthal), 
					torch.sin(inclination)*torch.cos(azimuthal),
					torch.zeros(azimuthal.shape, device=azimuthal.device)) ,dim=-1)
	return u1, u2 

def S2_exponential_map(omega, v):
	"""
	omega: torch.tensor .... x 3 is the expansion point on S2
	v: torch.tensor .... x 3 is the argument of the exponential map, i.e. vector in T(S2)
	"""
	v_norm = torch.norm(v, p=2, dim=-1)[...,None]
	return torch.cos(v_norm)*omega + torch.sin(v_norm)*(v/v_norm)

def S1_get_tangent_basis(angles):
	"""
	angles: n x 1 pytorch tensor 
	"""
	#vecs = torch.cat([torch.cos(angles), torch.sin(angles)], axis=1) ##euclidean coordinates of angle 
	tan_vecs = torch.cat([torch.sin(angles), torch.cos(angles)], axis=1)
	return tan_vecs 
