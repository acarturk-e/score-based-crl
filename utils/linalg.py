__all__ = ["subspace_intersection"]

from collections.abc import Sequence
import numpy as np
import numpy.typing as npt


def subspace_intersection(
    bases: Sequence[npt.NDArray[np.floating]],
) -> npt.NDArray[np.floating]:
    """Given an tuple of orthonormal subspace bases, returns an orthonormal basis of their intersection.

    Uses Zassenhaus algorithm from https://en.wikipedia.org/wiki/Zassenhaus_algorithm"""
    # TODO: atol is quite arbitrary here, it'd be better to use rtol
    assert len(bases) >= 1
    d = bases[0].shape[0]
    assert all([len(basis) == d for basis in bases])
    assert all(
        [np.allclose(np.eye(basis.shape[1]), basis.T @ basis) for basis in bases]
    )
    ATOL_ORTH = 1e-4
    basis = bases[0]
    for idx in range(1, len(bases)):
        # No intersection except point 0 possible: break
        if basis.shape[1] == 0:
            break
        zassenhaus_qr = np.linalg.qr(
            np.hstack(
                (
                    np.vstack((basis, basis)),
                    np.vstack((bases[idx], np.zeros_like(bases[idx]))),
                )
            ).T
        )
        basis = zassenhaus_qr.R[
            np.nonzero(np.abs(zassenhaus_qr.R[:, d - 1]) < ATOL_ORTH)[0], d:
        ].T
    return basis
