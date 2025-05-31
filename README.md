# Algorithms for Score-based Causal Representation Learning

This repository contains implementations of our algorithms in several papers on **score-based causal representation learning**. 

**Release notes**: This repo contains our multiple CRL algorithms from different papers. See the README files in specific folders for running each algorithm.
If you encounter any issues while using this code, please contact us, we would be happy to help. Earlier versions of some codes here were released in separate repositories that are no longer actively maintained. Please see the references below.

We use `conda` to manage dependencies. Please run
``conda env create -f environment.yml``
to create the conda environment `ScoreCRL` and install required packages.

## GSCALE-I (General Score-based Causal Latent Estimation via Interventions). 

### **Paper**: 
- [**Score-based Causal Representation Learning: Linear and General Transformations**](https://arxiv.org/abs/2402.00849) at JMLR 05/2025, *and*
- [**General Identifiability and Achievability for Causal Representation Learning**](https://arxiv.org/abs/2310.15450) at AISTATS 2024.

Notes:
- *Section 6: CRL under General Transformations* of the JMLR'2025 paper is an extended version of the AISTATS'2024 paper. The extended paper contains a much broader evaluation suite. 
- An outdated version of GSCALE-I code (for AISTATS'2024 paper) was [released earlier](https://github.com/bvarici/score-general-id-CRL). We recommend using *this* up-to-date repo.

**Setting**: General transformations ($$X = g(Z)$$) and single-node interventions (*two* per node)

We use GSCALE-I algorithm on several sub-settings as follows.

### GSCALE-I-images
Codes for the experiments on the image dataset (Section 7.3). Please see the detailed README file and the notebook within the corresponding folder.

### GSCALE-I-GLM
For the setting transform $g$ is set to a single-layer MLP, or a parameterized generalized linear model where $$X = \tanh(\mathbf{G} \cdot Z)$$ (see Section 7.2.1 of the paper). See the detailed README within the corresponding folder.

### GSCALE-I-MLP
For the setting transform $g$ is set to an MLP with tanh activations. Codes for the experiments with 3-layer ML (see Section 7.2.2 of the paper). See the detailed README file within the corresponding folder.

## LSCALE-I (Linear Score-based Causal Latent Estimation via Interventions). 

### **Paper**: Score-based Causal Representation Learning: Linear and General Transformations (https://arxiv.org/abs/2402.00849), published at JMLR (05/2025).

**Setting**: Linear transformations ($$X = \mathbf{G} \cdot Z$$) and single-node interventions (*one* per node)
 
- `l_scale_i.py` : contains (sub-)algorithms for LSCALE-I
- `l_scale_i_test.py` : main test file for running LSCALE-I with different settings
- `l_scale_i_analyze.py` : analysis of LSCALE-I results.


## UMNI-CRL (Unknown Multi-node Interventional CRL)

### **Paper**: [Linear Causal Representation Learning from Unknown Multi-node Interventions](https://arxiv.org/abs/2402.00849) at NeurIPS 2024

**Setting**: Linear transformations ($$X = \mathbf{G} \cdot Z$$) and unknown multi-node interventions

- `umni_crl.py` : contains (sub-)algorithms for UMNI-CRL
- `umni_crl_test.py` : main test file for running UMNI-CRL with different settings
- `umni_crl_analyze.py` : [INCOMPLETE] analysis of UMNI-CRL results

Note: The codebase was [released earlier](https://github.com/acarturk-e/umni-crl). We recommend using the current repo.


## Citation

If you find these algorithms helpful, please consider citing the corresponding paper(s).

**Bibtex**
```
@article{varici2024score,
  title={Score-based causal representation learning: Linear and General Transformations},
  author={Var{\i}c{\i}, Burak and Acart{\"u}rk, Emre and Shanmugam, Karthikeyan and Kumar, Abhishek and Tajer, Ali},
  journal={arXiv:2402.00849},
  year={2024},
}

@inproceedings{varici2024linear,
  title = {Linear Causal Representation Learning from Unknown Multi-node Interventions},
  author = {Var{\i}c{\i}, Burak and Acart{\"u}rk, Emre and Shanmugam, Karthikeyan and Tajer, Ali},
  booktitle = {Proc. Advances in Neural Information Processing Systems},
  year = {2024},
  month = {December},
  address = {Vancouver, Canada}
}


@inproceedings{varici2024general,
  title = 	 {General Identifiability and Achievability for Causal Representation Learning },
  author =       {Var{\i}c{\i}, Burak and Acart\"{u}rk, Emre and Shanmugam, Karthikeyan and Tajer, Ali},
  booktitle = 	 {Proc. International Conference on Artificial Intelligence and Statistics},
  year = 	 {2024},
  month = 	 {May},
  address = {Valencia, Spain}
}

@inproceedings{acarturk2024sample,
  title = {Sample Complexity of Interventional Causal Representation Learning},
  author = {Acart{\"u}rk, Emre and Var{\i}c{\i}, Burak and Shanmugam, Karthikeyan and Tajer, Ali},
  booktitle = {Proc. Advances in Neural Information Processing Systems},
  year = {2024},
  month = {December},
  address = {Vancouver, Canada}
}


@article{varici2023score,
  title={Score-based causal representation learning with interventions},
  author={Var{\i}c{\i}, Burak and Acartürk, Emre and Shanmugam, Karthikeyan and Kumar, Abhishek and Tajer, Ali},
  journal={arXiv:2301.08230},
  year={2023}
}
```
