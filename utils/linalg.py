__all__ = [
    "subspace_intersection",
    "subspace_intersection_from_cor",
    "orth_psd_matrix",
]

from collections.abc import Sequence
import numpy as np
import numpy.typing as npt

# Note: be careful that np.linalg.eigh returns eigenvalues in ascending order, but np.linalg.eigvals and np.linalg.svd returns in descending order, so it's really eigh inconsistency (scipy.linalg.eigh is also in descending order)

def subspace_intersection_from_cor(
    R_list: Sequence[npt.NDArray[np.floating]], tol: float = 1e-4
) -> npt.NDArray[np.floating]:
    """ directly work with correlation matrices
    first extract their orthonormal bases, then apply subspace intersection
    """
    bases = [orth_psd_matrix(R,tol) for R in R_list]
    return subspace_intersection(bases)


def subspace_intersection(
    bases: Sequence[npt.NDArray[np.floating]], tol: float = 1e-4
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
            np.nonzero(np.abs(zassenhaus_qr.R[:, d - 1]) < tol)[0], d:
        ].T
    return basis

def orth_psd_matrix(R: np.ndarray, tol: float = 1e-4) -> np.ndarray:
    """
    Given a square, symmetric, PSD matrix R, return an orthonormal basis 
    for its column space by thresholding eigenvalues below `tol`.
    """
    # Eigen-decomposition (R = Q * diag(eigvals) * Q.T)
    eigvals, eigvecs = np.linalg.eigh(R)
    # Filter out any eigenvalues <= tol
    mask = eigvals > tol
    # The eigenvectors associated with eigenvalues > tol span the column space
    Q = eigvecs[:, mask]
    # Columns of Q are already orthonormal if R is symmetric PSD
    return Q

