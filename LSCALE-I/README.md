## LSCALE-I (Linear Score-based Causal Latent Estimation via Interventions). 

### **Paper**: Score-based Causal Representation Learning: Linear and General Transformations (https://arxiv.org/abs/2402.00849) - published at JMLR

**Setting**: Linear transformations ($$X = \mathbf{G} \cdot Z$$) and single-node interventions (*one* per node)
 
- `l_scale_i.py` : contains (sub-)algorithms for LSCALE-I
- `l_scale_i_test.py` : main test file for running LSCALE-I with different settings
- `l_scale_i_analyze.py` : analysis of LSCALE-I results.

**Reproducing results**: Paper reports results for LSCALE-I algorithm on Tables 3, 4, 5, 6, 7, 8, 11. To replicate the results for any of the settings, adjust the parameters (e.g., `n`, `d`, `n_samples`, etc.) in `l_scale_i.py` accordingly.

At the end of an experiment, `l_scale_i.py` also outputs the analysis (e.g., MCC and SHD). To analyze saved results files later, check `l_scale_i_analyze.py` and commands therein (simply call `read_and_display_results(results_pickle_file_path)`)
