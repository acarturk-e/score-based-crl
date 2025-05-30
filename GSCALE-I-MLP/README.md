# GSCALE-I on vector data (with MLP transform)

### Paper: Score-based Causal Representation Learning: Linear and General Transformations
(https://arxiv.org/abs/2402.00849), published at JMLR 05/2025.


We use `conda` to manage dependencies. Please run ``conda env create -f environment.yml``
to create the conda environment and install required packages.

**Setting**: $X = g(Z)$ general transform, single-node hard interventions.

We consider $g$ that are 1 or 3-layer MLPs with tanh activation function. For instance, for 1-layer, we have $X = tanh(G.Z)$ (see Section 7.2 of the paper for details).

Note that for 1-layer case, we can more directly estimate the parameter matrix $G$ (refer to `g_scale_i_glm_tanh` function).

To run an end-to-end experiment once, simply run the bash script: `run_once.sh`. See `run_many_times.sh` to repeat the experiment for desired number of seeds.
Baseline setting generates data for a 3-layer MLP $g$ and MLP-based latent causal model. Adjust the setting as desired in `generate_data.py`.


### Files and structure

- `generate_data.py`: Latent sampling & latent-to-observed mapping.
- `run_score_diff`: Runs classification-based score difference estimation on generated data.
- `model_score_diff`: Modules for classification-based score difference estimation
- `run_gscalei`: Run GSCALE-I algorithm given the score differences.
- `analysis.py`: analyze a single run.
