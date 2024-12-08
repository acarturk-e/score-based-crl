# Algorithms for Score-based Causal Representation Learning

This repository contains implementations of the algorithms in our several papers on **score-based causal representation learning**. 

Note: We collect our multiple CRL algorithms from different papers here, so the overall repository needs some cleanup and enriched commentation. If you try to use any piece of this code and run into issues, please feel free to contact us, we would be happy to help. Earlier versions of some codes here were released at separate repos that are no longer actively maintained, please see the references below.


## UMNI-CRL (Unknown Multi-node Interventional CRL)

### **Paper**: Linear Causal Representation Learning from Unknown Multi-node Interventions (https://arxiv.org/abs/2402.00849) at NeurIPS 2024

**Setting**: Linear transformations ($$X = \mathbf{G} \cdot Z$$) and unknown multi-node interventions

- `umni_crl.py` : contains (sub-)algorithms for UMNI-CRL
- `umni_crl_test.py` : main test file for running UMNI-CRL with different settings
- `umni_crl_analyze.py` : [INCOMPLETE] analysis of UMNI-CRL results

Note: The codebase was released earlier at [https://github.com/acarturk-e/umni-crl](https://github.com/acarturk-e/umni-crl). We recommend using the current repo.

## LSCALE-I (Linear Score-based Causal Latent Estimation via Interventions). 

### **Paper**: Score-based Causal Representation Learning: Linear and General Transformations (https://arxiv.org/abs/2402.00849)

**Setting**: Linear transformations ($$X = \mathbf{G} \cdot Z$$) and single-node interventions (*one* per node)
 
- `l_scale_i.py` : contains (sub-)algorithms for LSCALE-I
- `l_scale_i_test.py` : main test file for running LSCALE-I with different settings
- `l_scale_i_analyze.py` : [INCOMPLETE] analysis of LSCALE-I results.


## GSCALE-I (General Score-based Causal Latent Estimation via Interventions). 

### **Paper**: 
- General Identifiability and Achievability for Causal Representation Learning (https://arxiv.org/abs/2310.15450) at AISTATS 2024, *and*
- Score-based Causal Representation Learning: Linear and General Transformations (https://arxiv.org/abs/2402.00849) 

Some notes:
- *Section 6: CRL under General Transformations* of the second paper is based on the first paper. 
- The first paper (AISTATS 2024) contains some preliminary experiments with GSCALE-I, so an outdated version of GSCALE-I code was released earlier at https://github.com/bvarici/score-general-id-CRL. We recommend using the current repo.
- The second paper (preprint) contains additional experiments.

**Setting**: General transformations ($$X = g(Z)$$) and single-node interventions (*two* per node)

In the AISTATS paper (Section 6.2) and the extended paper (Section 7.2), we consider a parameterized general linear model setting $$X = \tanh(\mathbf{G} \cdot Z)$$.
- `g_scale_i_algos.py`: contains GSCALE-I algorithm specialized to this tanh + linear transform setting.
- `g_scale_i_vector_test.py`: main test file for running this algorithm
- `g_scale_i_analyze.py`: [INCOMPLETE] analysis of the results

## GSCALE-I-images

### **Paper**: Score-based Causal Representation Learning: Linear and General Transformations (https://arxiv.org/abs/2402.00849)

Codes for the experiments on the image dataset (Section 7.3). Please see the detailed README file and the notebook within the corresponding folder.

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
