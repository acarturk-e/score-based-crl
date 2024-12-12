__all__ = ["StructuralCausalModel"]

from abc import ABC, abstractmethod
from typing import Any, Literal
import numpy as np


class StructuralCausalModel(ABC):
    """Abstract SCM. Subclasses need to implement ``_link_fn`` and ``_link_fn_grad``."""

    def __init__(
        self,
        n: int,
        fill_rate: float,
        randomize_top_order: bool = False,
        np_rng: np.random.Generator | None = None,
    ) -> None:
        self.n = n
        self.fill_rate = fill_rate

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

        self.adj_mat = np.triu(
            self.np_rng.random((self.n, self.n)) <= self.fill_rate,
            1,
        )
        self.adj_mat = self.adj_mat[self.top_order_inverse, :]
        self.adj_mat = self.adj_mat[:, self.top_order_inverse]
        self.pa = [np.nonzero(self.adj_mat[:, i])[0] for i in range(self.n)]
        self.ch = [np.nonzero(self.adj_mat[i, :])[0] for i in range(self.n)]

        self.variances = np_rng.random(self.n) + 0.5

    @abstractmethod
    def _link_fn(
        self,
        i: int,
        z_pa_i: np.ndarray[Any, np.dtype[np.floating]],
        mechanism: (Literal["obs"] | Literal["hard int"] | Literal["soft int"]) = "obs",
    ) -> np.ndarray[Any, np.dtype[np.floating]]:
        raise NotImplementedError("Abstract/pure virtual method!")

    @abstractmethod
    def _link_fn_grad(
        self,
        i: int,
        z_pa_i: np.ndarray[Any, np.dtype[np.floating]],
        mechanism: (Literal["obs"] | Literal["hard int"] | Literal["soft int"]) = "obs",
    ) -> np.ndarray[Any, np.dtype[np.floating]]:
        raise NotImplementedError("Abstract/pure virtual method!")

    def sample(
        self,
        shape: tuple[int],
        nodes_int: list[int] = [],
        type_int: Literal["hard int"] | Literal["soft int"] = "hard int",
        var_change_mech: Literal["increase", "scale"] = "scale",
        var_change: float = 0.1,
    ) -> np.ndarray[Any, np.dtype[np.floating]]:
        # Initialize independent Gaussian noises
        noises = self.np_rng.standard_normal(shape + (self.n, 1))
        for i in self.top_order:
            noises[..., i, :] *= (
                self.variances[i]
                if i not in nodes_int
                else (
                    self.variances[i] * var_change
                    if var_change_mech == "scale"
                    else self.variances[i] + var_change
                )
            ) ** 0.5
        # Form model samples from noise using topological order
        samples = np.zeros_like(noises)
        for i in self.top_order:
            link_fn_values = self._link_fn(
                i,
                samples[..., self.pa[i], :],
                mechanism="obs" if i not in nodes_int else type_int,
            )[..., 0]
            samples[..., i, :] = noises[..., i, :] + link_fn_values
        return samples

    def score_fn(
        self,
        samples: np.ndarray[Any, np.dtype[np.floating]],
        nodes_int: list[int] = [],
        type_int: Literal["hard int"] | Literal["soft int"] = "hard int",
        var_change_mech: Literal["increase", "scale"] = "scale",
        var_change: float = 0.1,
    ) -> np.ndarray[Any, np.dtype[np.floating]]:
        noises = np.zeros_like(samples)
        score_samples = np.zeros_like(samples)
        for i in self.top_order:
            link_fn_values = self._link_fn(
                i,
                samples[..., self.pa[i], :],
                mechanism="obs" if i not in nodes_int else type_int,
            )[..., 0]
            noises = samples[..., i, :] - link_fn_values
            score_of_mechs = -noises / (
                self.variances[i]
                if i not in nodes_int
                else (
                    self.variances[i] * var_change
                    if var_change_mech == "scale"
                    else self.variances[i] + var_change
                )
            )
            score_samples[..., i, :] += score_of_mechs
            score_samples[..., self.pa[i], :] -= score_of_mechs[
                ..., None
            ] * self._link_fn_grad(
                i,
                samples[..., self.pa[i], :],
                mechanism="obs" if i not in nodes_int else type_int,
            )
        return score_samples
