import numpy as np
import torch
from torch import linalg as LA
import sys 

PATH2SNEF = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "snefy-main"))
sys.path.append(PATH2SNEF)
from squared_neural_families.nets import kernels

class SquaredNNProd(torch.nn.Module):
	"""
	Defines a squared neural network multiplied by an appropriate measure.
	Args:
		D (int): num marginal products
		domain (str): 'Rd' for \mathbb{R}^d.
		measure (str): 'gauss' for standard Gaussian measure
		activation (str): 'cos' for cosine activations
		preprocessing (str): 'ident' for no preprocessing
		d (int): marginal dimension of the variable to be modelled
		n (int): Number of parameters in V
		dim (None or int): If not None, only model this index of the input.
			If not None, then d must be 1
		m (int): The number of rows in V, i.e. the width of the readout
			layer. If m is -1, parameterise by PD matrix V.T V
	Methods:
		integrate - Integrate the squared neural network against the measure.
			Optionally takes an extra_input, which could be the output of
			another neural network for conditional density estimation.
		forward - Forward pass through the squared network multiplied
			by the measure.
	"""
	def __init__(self, D, domain, measure, activation, preprocessing, d=2, n=100,
		dim = None, m=1, diagonal_V = False):
		super().__init__()
		self.D = D
		self.d = d
		self.n = n
		self.a = 1 #TODO: this is the parameter for snake activaitons.
		self.bound0 = None
		self.bound1 = None
		self.dim = dim
		self.measure = measure
		self.preprocessing = preprocessing

		self._initialise_params(d, n, m, diagonal_V) ## list of params 
		self._initialise_measure(measure) ## must be same for all marginals 
		self._initialise_activation(activation) ## must be same for all marginals (will use exp)
		self._initialise_kernel(domain, measure, activation, preprocessing) ## must be same for all marginals, preprocessing is ident

	def _initialise_activation(self, activation):
		if (activation == 'cos'):
			self.act = torch.cos
		elif (activation == 'cosmident'):
			self.act = lambda x: x - torch.cos(x)
		elif (activation == 'snake'):
			#self.act = lambda x: x + torch.sin(self.a*x)**2/self.a
			self.act = lambda x: x + (1-torch.cos(2*self.a*x))/(2*self.a)
		elif (activation == 'sin'):
			self.act = torch.sin
		elif (activation == 'relu'):
			self.act = torch.nn.ReLU()
			for B in self.Bs:
				B.requires_grad = False
		elif (activation == 'erf'):
			self.act = torch.erf
			for B in self.Bs:
				B.requires_grad = False
		elif (activation == 'exp'):
			self.act = torch.exp
			for B in self.Bs:
				B.requires_grad = False
		else:
			raise Exception("Unexpected activation.")

	def _initialise_measure(self, measure):
		if (measure == 'leb'):
			self.base_measure = None
			self.pdf = lambda x, log_scale: 0 if log_scale else 1
		elif (measure == 'uniformsphere'):
			# Reciprocal of surface area of sphere
			self.base_measure = None
			def pdf_func(x, log_scale):
				if log_scale:
					return self.D * (
						torch.lgamma(torch.tensor(self.d / 2)) 
						- (self.d / 2) * np.log(np.pi) 
						- np.log(2)
					)
				else:
					return (torch.exp(torch.lgamma(torch.tensor(self.d / 2)))
							/ (2 * np.pi**(self.d / 2)))**self.D
			self.pdf_ = pdf_func
			self.pdf = lambda x, log_scale: self.pdf_(x, log_scale).to(x.device)
		else:
			raise Exception("Unexpected measure.")

	def _initialise_params(self, d, n, m, diagonal_V):
		self.m = m

		self.Ws = torch.nn.ParameterList([torch.nn.Parameter(torch.from_numpy(np.random.normal(0, 1/np.sqrt(self.D), (n, d))*2).float()) for i in range(self.D)])

		if diagonal_V:
			assert (n == m)
			self.V = torch.nn.Parameter(torch.diag(torch.ones((n)).float()))
		else:
			if m == -1:
				size = n
			else:
				size = m
			V = np.random.normal(0, 1, (size, n))*np.sqrt(1/(n*size))
			self.V = torch.nn.Parameter(torch.from_numpy(V).float())
		if m == -1:
			self.initialise_vtv()

		self.Bs = torch.nn.ParameterList([torch.nn.Parameter(torch.from_numpy(np.zeros((n, 1))).float()) for i in range(self.D)])
		self.v0 = torch.nn.Parameter(torch.from_numpy(np.asarray([1.])).float())

	def initialise_vtv(self):
		self.m = -1
		self.VTV = torch.nn.Parameter((self.V.T @ self.V).data)

	def _initialise_kernel(self, domain, measure, activation, preprocessing):
		if (domain == 'Rd') and (measure == 'gauss') and (activation == 'cos')\
				and (preprocessing == 'ident'):
			name = 'cos'
		elif (domain == 'Rd') and (measure == 'gauss') and (activation == 'snake')\
				and (preprocessing == 'ident'):
			name = 'snake'
		elif (domain == 'Rd') and (measure == 'gauss') and (activation == 'cosmident')\
				and (preprocessing == 'ident'):
			name = 'cosmident'
		elif (domain == 'Rd') and (measure == 'gauss') and (activation == 'sin')\
				and (preprocessing == 'ident'):
			name = 'sin'
		elif (domain == 'Rd') and (measure == 'gauss') and (activation == 'relu')\
				and (preprocessing == 'ident'):
			name = 'arccos'
			for B in self.Bs:
				B.requires_grad = False
		elif (domain == 'Rd') and (measure == 'gauss') and (activation == 'erf')\
				and (preprocessing == 'ident'):
			name = 'arcsin'
			for B in self.Bs:
				B.requires_grad = False
		elif ((domain == 'Rd') and (measure == 'gauss') and (activation == 'exp')\
			and (preprocessing == 'sphere')) or \
			((domain == 'sphere') and (measure == 'uniformsphere') and \
			(activation == 'exp') and (preprocessing == 'ident')):
			name = 'vmf'
		elif (domain == 'sphere') and (measure == 'uniformsphere') and \
			(activation == 'relu') and (preprocessing == 'ident'):
			name = 'arccossphere'
			for B in self.Bs:
				B.requires_grad = False
		elif (type(domain) is list) and (measure == 'leb') and (activation == 'exp')\
			and (preprocessing == 'ident'):
			name = 'loglinear'
			self.bound0 = domain[0]; self.bound1 = domain[1]
			for B in self.Bs:
				B.requires_grad = False
		else:
			raise Exception("Unexpected integration parameters.")
		
		self.kernel = Kernel(name, self.a, self.bound0, self.bound1)

	def integrate(self, extra_input=0, log_scale=False):
		self.K = self._evaluate_kernel(extra_input, keep_dims=[])
		#    self.K = self.K * self.kernelt(self.Wt, self.Bt, extra_input)
		#VKV = self.V.T @ self.K @ self.V+ self.v0**2 ## <- m=1 case transpose
		#torch.vmap vectorises the operation. So we can do a batch trace on
		# (B, m, m) for B traces of mxm matrices
		#VKV = torch.vmap(torch.trace)(self.V @ self.K @ self.V.T)
		# Not available until very recent so we do something else instead
		#VKV = (self.V @ self.K @ self.V.T).diagonal(offset=0, dim1=-1, 
		#    dim2=-2).sum(-1).view((-1, 1, 1)) + self.v0**2
		VTV = self.VTV + 1e-3*torch.eye(self.VTV.shape[0],
			device = self.VTV.device)\
			if self.m == -1 else self.V.T @ self.V

		VKV = torch.sum(self.K * VTV, dim=[1,2]) + self.v0**2
		if log_scale:
			ret = torch.log(VKV)
		else:
			ret = VKV

		return ret

	def _evaluate_kernel(self, extra_input=0, keep_dims = []):
		t_param10 = self.Ws[0].view((self.Ws[0].shape[0], self.Ws[0].shape[1]))
		t_param20 = self.Bs[0].view((self.Bs[0].shape[0], 1))
		K = self.kernel(t_param10.contiguous(), 
				t_param20.contiguous(), extra_input)
		for i in range(1, self.D):
			t_param1i = self.Ws[i].view((self.Ws[i].shape[0], self.Ws[i].shape[1]))
			t_param2i = self.Bs[i].view((self.Bs[i].shape[0], 1))
			K = K * self.kernel(t_param1i.contiguous(), 
						t_param2i.contiguous(), extra_input)
		K = K.view((-1, self.n, self.n))
		return K

	def forward(self, y, extra_input=0, log_scale=False):     
		# If M is batch size, below is shape M x n
		feat = []
		for ds in range(self.D):
			feat.append(self.act(self.Ws[ds] @ y[...,(ds*self.d):(self.d*(ds+1))].T + self.Bs[ds]).T + extra_input)

		feat = torch.prod(torch.stack(feat), dim=0)
		if self.m == -1:
			# Batch matrix multiply of features gives shape M x n x n
			"""
			feat = feat.unsqueeze(2)
			Ktilde = torch.bmm(feat, torch.swapaxes(feat, 1, 2))
			VTV = self.VTV if self.m == -1 else self.V.T @ self.V
			squared_net = torch.sum(Ktilde*VTV, dim=[1,2]) + \
				self.v0**2
			"""
			VTV = self.VTV + 1e-3*torch.eye(self.VTV.shape[0],
				device = self.VTV.device)\
				if self.m == -1 else self.V.T @ self.V
			psiT_VTV = feat @ VTV
			squared_net = torch.bmm(psiT_VTV.view(-1, 1, self.n),
				feat.view(-1, self.n, 1)) + self.v0**2
		else:
			net_out = (self.V @ feat.T).T
			squared_net = torch.norm(net_out, dim=1)**2 + self.v0**2
			squared_net = squared_net.view((1, -1))
		if log_scale:
			logpdf = self.pdf(y, log_scale)
			return torch.log(squared_net) + logpdf

		pdf = self.pdf(y, log_scale)
		return squared_net*pdf

	def l2_lastlayer(self):
		VTV = self.VTV if self.m == -1 else self.V.T @ self.V
		return torch.norm(VTV)

class Kernel(torch.nn.Module):
	"""
	These are kernels used by the SquaredNN for closed-form integration.
	
	Args:
		name (str): Name of the kernel. 'cos'
	"""
	def __init__(self, name, a, bound0=None, bound1=None):
		super().__init__()
		self.a = a
		self.bound0 = bound0
		self.bound1 = bound1
		self._init_kernel(name)

	def _init_kernel(self, name):
		if name == 'cos':
			self.kernel = lambda W, B, extra_input: \
				kernels.cos_kernel(W, W, B+extra_input, B+extra_input)
		elif name == 'snake':
			self.kernel = lambda W, B, extra_input: \
				kernels.snake_kernel(W, W, B+extra_input, B+extra_input,
						a=self.a)
		elif name == 'cosmident':
			self.kernel = lambda W, B, extra_input: \
				kernels.cos_minus_ident_kernel(W, W, B+extra_input, B+extra_input)
		elif name == 'sin':
			self.kernel = lambda W, B, extra_input: \
				kernels.sin_kernel(W, W, B+extra_input, B+extra_input)
		elif name == 'arccos':
			self.kernel = lambda W, B, extra_input: \
				kernels.arc_cosine_kernel(W, W, B+extra_input, B+extra_input)
		elif name == 'arccossphere':
			self.kernel = lambda W, B, extra_input: \
				kernels.arc_cosine_kernel_sphere\
				(W, W, B+extra_input, B+extra_input)
		elif name == 'arcsin':
			self.kernel = lambda W, B, extra_input: \
				kernels.arc_sine_kernel(W, W, B+extra_input, B+extra_input)
		elif name == 'vmf':
			self.kernel = lambda W, B, extra_input: \
				kernels.vmf_kernel(W, W, B+extra_input, B+extra_input)
		elif name == 'loglinear':
			self.kernel = lambda W, B, extra_input: \
				kernels.log_linear_kernel(W, W, B+extra_input, B+extra_input,
						a=self.bound0, b=self.bound1)
		elif name == 'relu1d':
			self.kernel = lambda W, B, extra_input:\
				kernels.relu1d_kernel(W, W, B+extra_input, B+extra_input,
					a=self.a)
		else:
			raise Exception("Unexpected kernel name.")

	def forward(self, W,  B, extra_input=0):
		return self.kernel(W, B, extra_input)


"""
Miscellaneous functions used in the above implementations
"""
def normpdf(val, std=1, log_scale = False):
	if log_scale == True:
		#ret = torch.sum(\
		#    -0.5*torch.log(2*np.pi)-0.5*val**2/std**2\
		#    -torch.log(std),dim=1)
		ret = torch.sum(-0.5*val**2/std**2, dim=1)\
			-0.5*np.log(2*np.pi)*val.shape[1]\
			-torch.log(std)*val.shape[1]
	else:
		ret = torch.prod((1/torch.sqrt(2*np.pi*std**2))*\
			torch.exp(-0.5*val**2/std**2), dim=1)
	return ret




