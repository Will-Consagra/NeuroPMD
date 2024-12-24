Code for deep neural network density estimation on product Riemannian manifold domains (NeuroPMD). 
The project includes the network architecture, training routines, and scripts to reproduce key simulation results presented in the paper. 
The code is available on the project GitHub page: https://github.com/Will-Consagra/NeuroPMD.

# Setting Up the Environment

The conda environment (concon_ax) is defined in the environment.yml file. To create and activate the environment, use the following commands:

`conda env create -f environment.yml` 
`conda activate concon_ax`

# Repository Structure

- `neuroPMD/` contains pytorch code implementing our network architecture

- `optimization/` contains code for training the network 

- `simulations/` contains code for reproducing some simulation results from the paper 

# Running Simulations

To run the simulation for the torus case in Section 5.1 of the paper, run

`python simulations/MC_sim_study_T2.py mc_exp --data_dir simulations/T2_data --out_dir simulations/T2_data/T2_results`

Replace mc_exp with values ranging from 1 to 21 for the different experimental runs.

Once the simulations are completed, you can compile and evaluate the results using the following script:

`python evaluation.py  --data_dir simulations/T2_data/T2_results --Nexp 20 --metric AE (or Normalized_L2_error)`

This script prints the MC simulation results for the T^2 case directly to the screen. 

