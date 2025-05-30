# GSCALE-I on image data

### Paper: Score-based Causal Representation Learning: Linear and General Transformations
(https://arxiv.org/abs/2402.00849), published at JMLR 05/2025. See Section 7.5 for experiments on image data.

We use `conda` to manage dependencies. Please run
``conda env create -f environment.yml``
to create the conda environment and install required packages.
The "installed" environment name is important for scripts.

To run an end-to-end experiment, simply run the bash scripts.
`run_5_times.sh` is the baseline setting: It generates image data
for 3 balls data and trains a two-step autoencoder 5 times.


TODO: analysis files are not cleaned-up.

### Files and structure

- `generate_data.py`: Latent sampling & image rendering
- `cdsde.py`: Classification-based score difference estimation
    structure
- `train_ldr.py`: Estimate log-density ratio & score difference
    from image data
- `train_reconstruct.py`: Train an autoencoder from images with
    bottleneck dim 64 (default)
- `x_to_y.py`: Map data (image samples & score differences) to
    output domain of the step 1 encoder
- `train_disentangle.py`: Train an autoencoder from step 1 encoder
    output subject to "main loss": Aims to disentangle the learned
    representations using the score difference-based loss

### On disentanglement loss

Here is the data flow.
1. **Data generation.** Generate $n$ dimensional $Z$ according to some causal latent distribution. Then, map latent $Z$ to images, $X$, through rendering engine $g \colon \mathbb{R}^{n} \to [0, 255]^{64 \times 64 \times 3}$.
3. **Learn score differences in image domain.** Score difference is defined as $\mathbf{d}\_{X}^{i}(x) \coloneqq \nabla\_{\mathbf{x}} \log \frac{p\_{X}^{i}(\mathbf{x})}{p\_{X}(\mathbf{x})}$. Both input and output shapes are images!
4. **Autoencoder step 1 (reconstruct).** Reduce images to an intermediate vector $Y \in \mathbb{R}^{64}$ through encoder-decoder pair $h_{1}, h_{1}^{-1}$ with bottleneck dimension 64 while ensuring perfect image reconstruction, i.e., while minimizing $$h\_{1}, h^{-1}\_{1} \gets \arg\min_{h, h^{-1}} \mathbb{E} \big\|h^{-1}(h(X)) - X\big\|\_{2}^{2}  .$$ Finally, transform both images and score difference functions to this 64-dimensional domain, i.e., $$Y = h(X) , \qquad\mathbf{d}\_{Y}^{i}(y) = \mathbf{J}\_{h^{-1}}^{\top}(y) \cdot \mathbf{d}_{X}^{i}(x).$$
6. **Autoencoder step 2 (disentangle).** Learn a final autoencoder pair $h_{2}, h_{2}^{-1}$ subject to reconstruction **and** disentanglement constraints. To do so, first define a score change matrix $$\mathbf{D}\_{i}(h, h^{-1}) \coloneqq \mathbb{E} \big\|\mathbf{J}\_{h^{-1}}^{\top}(\hat z) \cdot \mathbf{d}\_{Y}^{i}(y)\big\|\_{1}.$$ Then, learn an autoencoder that minimizes a disentanglement term plus the reconstruction loss, i.e.,
$$h_{2}, h_{2}^{-1} = \arg\min_{h, h^{-1}} \sum_{i = 1}^{n} \big\|\mathbf{D}\_{i}(h, h^{-1}) - \mathbf{e}\_{i}\big\|\_{1} + \mathbb{E} \big\|h^{-1}(h(Y)) - Y\big\|\_{2}^{2}\,.$$ The overall autoencoder that estimates the latent variables is $(h_{2} \circ h_{1}, h_{1}^{-1} \circ h_{2}^{-1})$, i.e., $\hat{Z} = h_{2}(h_{1}(X))$.
