import pickle 
import numpy as np 
import os 
from scipy.stats import sem, qmc
import torch 
import argparse

def mc_TD_sampler(Q, D, scramble=True):
    sampler = qmc.Sobol(d=D, scramble=scramble)
    mc_int_points = torch.from_numpy(sampler.random_base2(int(np.log2(Q)))).float()
    mc_points = (2*torch.pi * mc_int_points - torch.pi)
    return mc_points

# Adding optional arguments
parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, default="T2_data/T2_results",
                    help="Directory where data is stored (default: T2_data/T2_results)")
parser.add_argument("--Nexp", type=int, default=20,
                    help="Number of experiments (default: 20)")
parser.add_argument("--metric", type=str, default="AE",
                    choices=["AE", "L2_error", "hell_loss", "l2_corr", "Normalized_L2_error", "time"],
                    help='Evalion metric to be used (default: AE)')

args = parser.parse_args()

data_dir = args.data_dir
Nexp = args.Nexp
metric = args.metric

inif = []
for i in range(1,Nexp+1):
	fname_i = os.path.join(data_dir, "results_inif_%s.pkl"%i)
	if os.path.exists(fname_i):	
		with open(fname_i, "rb") as pklfile:
			data = pickle.load(pklfile)
		inif.append(data[metric])

if len(inif):
	print("----NeuroPMD Results----")
	print("Mean", np.mean(inif))
	print("SE", sem(inif))

KDE = []
for i in range(1,Nexp+1):
	fname_i = os.path.join(data_dir, "results_kde_%s.pkl"%i)
	if os.path.exists(fname_i):	
		with open(fname_i, "rb") as pklfile:
			data = pickle.load(pklfile)
		KDE.append(data[metric])

if len(KDE):
	print("----KDE Results----")
	print("Mean", np.mean(KDE))
	print("SE", sem(KDE))

TPB = []
for i in range(1,Nexp+1):
	fname_i = os.path.join(data_dir, "results_tpb_%s.pkl"%i)
	if os.path.exists(fname_i):
		with open(fname_i, "rb") as pklfile:
			data = pickle.load(pklfile)
		lam2s = list(data.keys())
		EISE = [data[lam2]["Approx_ISE"] for lam2 in lam2s]
		ix_min = EISE.index(min(EISE))
		TPB.append(data[lam2s[ix_min]][metric])

if len(TPB):
	print("----TPB Results----")
	print("Mean", np.mean(TPB))
	print("SE", sem(TPB))


pSNF = []
for i in range(1,Nexp+1):
	fname_i = os.path.join(data_dir, "results_snef_%s.pkl"%i)
	if os.path.exists(fname_i):	
		with open(fname_i, "rb") as pklfile:
			data = pickle.load(pklfile)
		pSNF.append(data[metric])

if len(pSNF):
	print("----pSNF Results----")
	print("Mean", np.mean(pSNF))
	print("SE", sem(pSNF))
