__all__ = ["QuadraticSCM"]

from typing import Any, Literal
import numpy as np

from . import StructuralCausalModel


class QuadraticSCM(StructuralCausalModel):
    def __init__(
        self,
        n: int,
        fill_rate: float,
        randomize_top_order: bool = False,
        link_mat_min_sv: float = 1e-1,
        np_rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(n, fill_rate, randomize_top_order, np_rng)

        self._link_mats = [
            np.empty((len(self.pa[i]),), dtype=np.floating) for i in range(self.n)
        ]
        for i in range(self.n):
            # TODO: Here, I enforce the link matrices to be full rank.
            # This may be stricter than we should be enforcing.
            k = len(self.pa[i])
            lmi_sqrt = 2.0 * self.np_rng.random((k, k)) - 1.0
            if k != 0:
                lmi_sqrt_svs = np.linalg.svd(lmi_sqrt, compute_uv=False)
                while lmi_sqrt_svs[-1] < link_mat_min_sv:
                    lmi_sqrt = 2.0 * self.np_rng.random((k, k)) - 1.0
                    lmi_sqrt_svs = np.linalg.svd(lmi_sqrt, compute_uv=False)
            self._link_mats[i] = lmi_sqrt.T @ lmi_sqrt

    def _link_fn(
        self,
        i: int,
        z_pa_i: np.ndarray[Any, np.dtype[np.floating]],
        mechanism: (Literal["obs"] | Literal["hard int"] | Literal["soft int"]) = "obs",
    ) -> np.ndarray[Any, np.dtype[np.floating]]:
        if mechanism == "obs":
            return np.sqrt(np.swapaxes(z_pa_i, -1, -2) @ self._link_mats[i] @ z_pa_i)
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
            return (self._link_mats[i] @ z_pa_i) / np.sqrt(
                np.swapaxes(z_pa_i, -1, -2) @ self._link_mats[i] @ z_pa_i
            )
        elif mechanism == "hard int":
            return np.zeros(z_pa_i.shape[:-2] + (1, 1))
        if mechanism == "soft int":
            return 0.5 * self._link_fn_grad(i, z_pa_i, "obs")
