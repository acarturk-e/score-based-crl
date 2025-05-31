# GSCALE-I on vector data: GLM

### Paper: Score-based Causal Representation Learning: Linear and General Transformations
(https://arxiv.org/abs/2402.00849), published at JMLR 05/2025.


We use `conda` to manage dependencies. Please run ``conda env create -f environment.yml``
to create the conda environment and install required packages.

**Setting**: In this generalized linear model case, we consider 1-layer MLP with tanh activation, $$\mathbf{X} = \tanh(\mathbf{G} \cdot \mathbf{Z})$$ with 2 single-node hard interventions per node.

- `g_scale_i_algos.py`: Contains the main function `g_scale_i_glm_tanh`
- `g_scale_i_vector_test.py`: Runs the experiments. Adjust the parameters latent dimension $n$, observed dimension $d$ and others as desired. (used for the experiments reported in Table 9 of the paper.)
- `g_scale_i_analysis.py`: Reads and displays the saved results.

