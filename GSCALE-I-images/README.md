# GSCALE-I on image data

We use `conda` to manage dependencies. Please run
``conda env create -f environment.yml``
to create the conda environment and install required packages.

To run an end-to-end experiment, simply run the bash scripts.
`run_5_times_2step_cpu.sh` is the baseline setting: It
generates relevant data and trains a  two-step autoencoder
training 5 times.

You can run any script that ends with `_cpu.sh` on any computer.
Scripts ending with `_cci.sh` are designed for an internal GPU
cluster and may not be immediately portable.
