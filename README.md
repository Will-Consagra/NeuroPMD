Code for deep neural network density estimation on product Riemannian manifold domains (NeuroPMD). 
The project includes the network architecture, training routines, and scripts to reproduce key results presented in the paper. 
The code is available on the project GitHub page: https://github.com/Will-Consagra/NeuroPMD.

# Setting Up the Environment

The conda environment (concon_ax) is defined in the environment.yml file. To create and activate the environment, use the following commands:

`conda env create -f environment.yml` 

`conda activate concon_ax`

Alternatively, you can use a prebuilt Docker image for this environment here `https://hub.docker.com/r/zhengwustat/concon_ax_env`. 

See a setup example here: `https://guzhiling.github.io/not-so-prime/docs/Containers.html`.

If you wish to run the connectomics visualization script `vizSC.py`, you'll need to also create the environment (viz_env)

`conda env create -f viz_environment.yml` 

# Repository Structure

- `neuroPMD/` contains pytorch code implementing our network architecture

- `optimization/` contains code for training the network 

- `simulations/` contains code for reproducing simulation results from the paper 

- `realdata/` contains code for reproducing real data results from the paper 

# Running Simulations

To run the simulations in Section 5.1 of the paper, using the `concon_ax` environment run:

For T2

`python simulations/MC_sim_study_T2.py mc_exp --data_dir simulations/T2_data --out_dir simulations/T2_data/T2_results`

For T4 **(WARNING: Run on GPU cluster to avoid excessively long computing time)**

`python simulations/MC_sim_study_T4.py mc_exp --data_dir simulations/T4_data --out_dir simulations/T4_data/T4_results`

Replace mc_exp with values ranging from 1 to 21 for the different experimental runs.

Once the simulations are completed, you can compile and evaluate the results using the following script:

`python evaluation.py  --data_dir simulations/T2_data/T2_results --Nexp 20 --metric AE (or Normalized_L2_error)`

and 

`python evaluation.py  --data_dir simulations/T4_data/T4_results --Nexp 20 --metric AE (or Normalized_L2_error)`

which prints the MC simulation results directly to the screen. 

# Running Connectomics Example 

To fit a NeuroPMD to the structural connectivity example data from Section 5.2 of the paper, from the head directory and using the `concon_ax` environment run:

`python realdata/fitSC.py --device_num 0 --lambda_2 0.0001 --max_degree 10 --rank 256 --depth 6 --cyclic --viz --cp --data_dir realdata/endpoints/10`

**(WARNING: Run the above command on a GPU to avoid excessively long computing times)**

To visualize the VTK files of the marginal connectivities, from the head directory and using the `viz_env` environment run:

`python realdata/vizSC.py --marg_surf_f realdata/endpoints/10/figures/fmodel_degree_10_width_256_depth_6_w0_10_lam_0.0001/f_evals_marg_mean10000.pkl \\
				--SCFILE realdata/endpoints/10/figures/fmodel_degree_10_width_256_depth_6_w0_10_lam_0.0001/CC_lh_ico420000.npy \\
				--OUTDIR realdata/endpoints/10/figures/fmodel_degree_10_width_256_depth_6_w0_10_lam_0.0001 \\
				--endpoint_file realdata/endpoints/10/LH__points_euc.pt
`

The .vtk files showing the marginal connectivity from several ROI's will be written to the sub folder `realdata/endpoints/10/figures/fmodel_degree_10_width_256_depth_6_w0_10_lam_0.0001`.

