__all__ = ["LinearSCM"]

from typing import Any, Literal
import numpy as np

from . import StructuralCausalModel


class LinearSCM(StructuralCausalModel):
    def __init__(
        self,
        n: int,
        fill_rate: float,
        randomize_top_order: bool = False,
        np_rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(n, fill_rate, randomize_top_order, np_rng)

        self._link_vector = [
            np.empty((len(self.pa[i]),), dtype=np.floating) for i in range(self.n)
        ]
        for i in range(self.n):
            k = len(self.pa[i])
            self._link_vector[i] = np.sign(self.np_rng.random((1, k)) - 0.5) * (0.5 + self.np_rng.random((1, k)))

        self._weight_mat = np.zeros((self.n, self.n))
        for i in self.top_order:
            self._weight_mat[i, self.pa[i]] = self._link_vector[i]

    def _link_fn(
        self,
        i: int,
        z_pa_i: np.ndarray[Any, np.dtype[np.floating]],
        mechanism: (Literal["obs"] | Literal["hard int"] | Literal["soft int"]) = "obs",
    ) -> np.ndarray[Any, np.dtype[np.floating]]:
        if mechanism == "obs":
            return self._link_vector[i] @ z_pa_i
        elif mechanism == "hard int":
            return np.zeros(z_pa_i.shape[:-2] + (1, 1))
        if mechanism == "soft int":
            return 0.5 * self._link_fn(i, z_pa_i, "obs")

    def _link_fn_grad(
        self,
        i: int,
        z_pa_i: np.ndarray[Any, np.dtype[np.floating]],
        mechanism: (Literal["obs"] | Literal["hard int"] | Literal["soft int"]) = "obs",
    ) -> np.ndarray[Any, np.dtype[np.floating]]:
        if mechanism == "obs":
            return self._link_vector[i].T
        elif mechanism == "hard int":
            return np.zeros(z_pa_i.shape[:-2] + (1, 1))
        if mechanism == "soft int":
            return 0.5 * self._link_fn_grad(i, z_pa_i, "obs")
