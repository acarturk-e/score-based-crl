__all__ = ["MlpSCM"]

from typing import Literal
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from . import StructuralCausalModel


class Mlp(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        if input_dim != 0:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_dim != 0:
            return self.fc2(torch.relu(self.fc1(x))).squeeze(-1)
        return torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)


class MlpSCM(StructuralCausalModel):
    def __init__(
        self,
        n: int,
        fill_rate: float,
        randomize_top_order: bool = False,
        np_rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(n, fill_rate, randomize_top_order, np_rng)
        self._obs_link_fns = [Mlp(len(self.pa[i]), 32, 1) for i in range(self.n)]

    def _link_fn(
        self,
        i: int,
        z_pa_i: npt.NDArray[np.floating],
        mechanism: Literal["obs"] | Literal["hard int"] | Literal["soft int"] = "obs",
    ) -> npt.NDArray[np.floating]:
        if mechanism == "obs":
            return self._obs_link_fns[i](torch.from_numpy(z_pa_i).float().squeeze(-1)).unsqueeze(-1).unsqueeze(-1).numpy()
        elif mechanism == "hard int":
            return np.zeros(z_pa_i.shape[:-2] + (1, 1))
        if mechanism == "soft int":
            return 0.5 * self._link_fn(i, z_pa_i, "obs")

    def _link_fn_grad(
        self,
        i: int,
        z_pa_i: npt.NDArray[np.floating],
        mechanism: Literal["obs"] | Literal["hard int"] | Literal["soft int"] = "obs",
    ) -> npt.NDArray[np.floating]:
        if mechanism == "obs":
            return torch.func.vmap(torch.func.jacfwd(self._obs_link_fns[i]))(torch.from_numpy(z_pa_i).float().squeeze(-1)).unsqueeze(-1).numpy()
            #return torch.func.jacfwd(self._obs_link_fns[i])(torch.from_numpy(z_pa_i).float().squeeze(-1)).numpy()
        elif mechanism == "hard int":
            return np.zeros(z_pa_i.shape[:-2] + (1, 1))
        if mechanism == "soft int":
            return 0.5 * self._link_fn_grad(i, z_pa_i, "obs")
