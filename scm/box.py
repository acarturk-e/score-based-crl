__all__ = ["BoxSCM"]

from typing import Any, Literal

import numpy as np


class BoxSCM:
    """Truncated linear Gausian structural causal model

    **IMPORTANT NOTE:** This class is NOT a subclass of
    `scm.StructuralCausalModel` since (a) it doesn't implement the score
    function methods and (b) its noise model is not generic Gaussian.

    ### Graph model
    Sample all 'valid' edges i.i.d. from Bernoulli(`fill_rate`).

    ### Weight model
    Weights are sampled from Uniform(+-[0.25, 1]).

    ### Noise model
    Noises are zero mean independent Gaussian with variances sampled
    independently across nodes from Uniform([0.01, 0.02]).

    ### Truncated sampling procedure
    Samples are truncated to `[-box_size, box_size]` in all axes.
    Sampling is done via rejection sampling."""
    def __init__(
        self,
        n: int,
        fill_rate: float,
        box_size: float = 1.0,
        randomize_top_order: bool = True,
        np_rng: np.random.Generator | None = None,
    ) -> None:
        assert n > 0
        self.n = n
        self.fill_rate = fill_rate

        self.box_size = box_size

        if np_rng is None:
            np_rng = np.random.default_rng()
        self.np_rng = np_rng

        if randomize_top_order:
            top_order = np_rng.permutation(self.n)
        else:
            top_order = np.arange(self.n, dtype=np.int64)
        self.top_order = top_order
        self.top_order_inverse = np.arange(self.n)
        self.top_order_inverse[self.top_order] = np.arange(self.n)

        adj_mat = np.triu(
            self.np_rng.random((self.n, self.n)) <= self.fill_rate, 1)

        self.adj_mat = adj_mat[self.top_order_inverse, :][:, self.top_order_inverse]
        self.pa = [np.nonzero(self.adj_mat[:, i])[0] for i in range(self.n)]
        self.ch = [np.nonzero(self.adj_mat[i, :])[0] for i in range(self.n)]

        self.variances = 0.01 + 0.01 * np_rng.random(self.n)

        self.weight_vectors = [
            np.empty((len(self.pa[i]),)) for i in range(self.n)
        ]
        for i in range(self.n):
            k = len(self.pa[i])
            self.weight_vectors[i] = (
                np.sign(self.np_rng.random((k,)) - 0.5) *
                (0.25 + 0.75 * self.np_rng.random((k,))))


    def _link_fn(
        self,
        i: int,
        z_pa_i: np.ndarray[Any, np.dtype[np.floating]],
        mechanism: (Literal["obs"] | Literal["hard int"] | Literal["soft int"]) = "obs",
    ) -> np.ndarray[Any, np.dtype[np.floating]]:
        if mechanism == "obs":
            return self.weight_vectors[i] @ z_pa_i
        elif mechanism == "hard int":
            return np.zeros(z_pa_i.shape[:-2] + (1, 1))
        if mechanism == "soft int":
            return 0.5 * self._link_fn(i, z_pa_i, "obs")


    def sample(
        self,
        shape: tuple[int],
        nodes_int: list[int] = [],
        type_int: Literal["hard int"] | Literal["soft int"] = "hard int",
        var_change_mech: Literal["increase", "scale"] = "scale",
        var_change: float = 0.1,
    ) -> np.ndarray[Any, np.dtype[np.floating]]:
        assert len(shape) == 1
        # Initialize independent Gaussian noises
        # Form model samples from noise using topological order
        samples = np.zeros(shape + (self.n, 1))
        for i in self.top_order:
            link_fns_i = self._link_fn(
                i,
                samples[..., self.pa[i], :],
                mechanism="obs" if i not in nodes_int else type_int)
            samples_i = np.zeros((0, 1))
            while samples_i.shape[0] < shape[0]:
                noises_i = self.np_rng.standard_normal(link_fns_i.shape) * (
                    self.variances[i]
                    if i not in nodes_int
                    else (
                        self.variances[i] * var_change
                        if var_change_mech == "scale"
                        else self.variances[i] + var_change
                    )
                ) ** 0.5
                accepted_i = link_fns_i + noises_i
                accepted_mask = np.abs(accepted_i) <= self.box_size
                accepted_i = accepted_i[accepted_mask, None]
                link_fns_i = link_fns_i[~accepted_mask, None]
                samples_i = np.concatenate((samples_i, accepted_i), axis=0)
            samples[..., i, :] = samples_i[:shape[0], :]
        return samples


### Test
if __name__ == "__main__":
    n = 6
    degree = 2
    fill_rate = n * degree / ((n * (n - 1)) / 2)
    box_size = 1.0
    n_samples = 100
    scm = BoxSCM(n, fill_rate, box_size=box_size)
    samples = scm.sample((n_samples,))
    assert samples.shape == (n_samples, n, 1)
    assert np.all(np.abs(samples) <= box_size)
    print(samples[:10, :, 0])
