__all__ = ["Renderer", "mcc", "shd"]

import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Circle


def _generate_colors(
    n: int,
    saturation: float = 0.9,
    value: float = 0.9,
    use_int: bool = False,
) -> npt.NDArray:
    """Generates RGB colors with equally spaced hues in HSV model.

    Sets
    - Hues equally spaced starting from `0.0`
    - Saturation to `saturation`, default `0.9`
    - Value to `value`, default `0.9`

    Output format is `[0.0, 1.0)` if `use_int` is `False`
    and `[0, 255]` if it is `True`."""
    hsvs = np.zeros((n, 3))
    hsvs[:, 0] = np.linspace(0.0, 1.0, n + 1)[:-1]  # Hue
    hsvs[:, 1] = saturation
    hsvs[:, 2] = value
    rgbs = matplotlib.colors.hsv_to_rgb(hsvs)
    if use_int:
        rgbs = (rgbs * 255.0).astype(int)
    return rgbs


class Renderer:
    def __init__(
        self,
        num_balls: int = 3,
        scr_width: int = 64,
        scr_height: int = 64,
        ball_radius: int = 8,
        rng: np.random.Generator | None = None,
    ):
        if scr_width != scr_height:
            raise NotImplementedError("Width != height case not implemented yet")

        self.scr_width = scr_width
        self.scr_height = scr_height
        self.ball_radius = ball_radius
        self.num_balls = num_balls

        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

        self.ball_colors = [
            tuple(color)
            for color in _generate_colors(num_balls)]


    def render_n_balls(
        self,
        zs: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.uint8]:
        """Draws `num_balls` balls with locations given in `zs` and returns
        the RGB (`uint8`) bitmap

        Convention: In every sample `z` in `zs`, the x-axis of ball `i`
        is given by `z[2*i]` and its y-axis is given by `z[2*i+1]`."""
        assert zs.shape[-1] == 2 * self.num_balls
        assert np.all(zs >= 0.0) and np.all(zs <= 1.0)

        # Normalize the margins for printing
        zs = (
            (self.ball_radius / self.scr_width) +
            (1.0 - 2.0 * self.ball_radius / self.scr_width) * zs)

        # Process the batch one by one
        xs = np.empty(zs.shape[:-1] + (self.scr_width, self.scr_height, 3), dtype=np.uint8)
        ci = 0
        for index in np.ndindex(zs.shape[:-1]):
            if ci % 1000 == 0:
                print(f"{ci = }")
            ci += 1

            fig = Figure(figsize=(self.scr_width, self.scr_height), dpi=1)
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot()

            # Draw circles
            for ball_idx in range(self.num_balls):
                circle = Circle(
                    (zs[index + (2 * ball_idx,)], zs[index + (2 * ball_idx + 1,)]),
                    radius=self.ball_radius / self.scr_width,
                    color=self.ball_colors[ball_idx])

                ax.add_artist(circle)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal', 'box')
            ax.set_xticks([])
            ax.set_yticks([])

            canvas.draw()
            rgba = np.asarray(canvas.buffer_rgba())
            xs[index] = rgba[:, :, :3]

            fig.clear()

        return xs


def mcc(x_est: npt.NDArray[np.floating], x_gt: npt.NDArray[np.floating]) -> float:
    """Computes mean correlation coefficient between `x_est` and `x_gt`

    Data dimension: `(n_samples, n)`. Computes the correlation coefficients
    between entries of `x_est` and `x_gt`, and solves the maximum linear sum
    assignment problem."""
    from scipy.optimize import linear_sum_assignment  # type: ignore
    n_samples, n = x_gt.shape
    abs_corr = np.abs(np.corrcoef(x_gt, x_est, rowvar=False)[n:, :n])
    row_ind, col_ind = linear_sum_assignment(abs_corr, maximize=True)
    return abs_corr[row_ind, col_ind].mean()


def shd(g_est: npt.NDArray[np.bool], g_gt: npt.NDArray[np.bool]) -> int:
    """Computes structural Hamming distance between `g_est` and `g_gt`"""
    assert g_est.ndim == 2 and g_est.shape == g_gt.shape
    n = g_est.shape[0]
    return (
        ((g_est | np.eye(n, dtype=bool)) ^ (g_gt | np.eye(n, dtype=bool)))
        .astype(int)
        .sum()
    )


# TEST:
if __name__ == "__main__":

    def test_generate_colors():
        # Example: generate 10 colors
        colors = _generate_colors(20)

        # Plot the colors to visualize
        fig, ax = plt.subplots(figsize=(8, 2))
        for i, color in enumerate(colors):
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))  # type: ignore
        ax.set_xlim(0, len(colors))
        ax.set_ylim(0, 1)
        plt.axis("off")
        plt.show()

    test_generate_colors()
