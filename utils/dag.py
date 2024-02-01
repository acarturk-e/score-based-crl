"""Directed Acyclic Graph (DAG) utilities

Notes on notation.
Let G be a graph on n nodes with adjacency matrix A. Then, A[i, j] = True <=> i -> j in G.
This means that parents of node i in G are exactly non-zero entries in i-th column of A.
"""
from typing import Any
import numpy as np


def topological_order(
    adj_mat: np.ndarray[Any, np.dtype[np.bool_]],
) -> np.ndarray[Any, np.dtype[np.intp]] | None:
    (n, _) = adj_mat.shape
    subgraph = np.ma.MaskedArray[Any, np.dtype[np.bool_]](adj_mat)
    top_order = np.arange(n)
    for i in range(n):
        pa_cnts = np.count_nonzero(subgraph, axis=0)
        possible_root = np.argmin(pa_cnts)
        if pa_cnts[possible_root] != 0:
            return None
        top_order[i] = possible_root
        subgraph[:, possible_root] = np.ma.masked
        subgraph[possible_root, :] = np.ma.masked
    return top_order


def transitive_closure(
    adj_mat: np.ndarray[Any, np.dtype[np.bool_]],
) -> np.ndarray[Any, np.dtype[np.bool_]] | None:
    """Returns `None` if input is not a DAG"""
    tr_closure = np.zeros_like(adj_mat)
    top_order = topological_order(adj_mat)
    if top_order is None:
        return None
    for i in top_order:
        pa_i = adj_mat[:, i].nonzero()[0]
        tr_closure[:, i] |= adj_mat[:, i]
        for j in pa_i:
            tr_closure[:, i] |= tr_closure[:, j]
    return tr_closure


def transitive_reduction(
    adj_mat: np.ndarray[Any, np.dtype[np.bool_]],
) -> np.ndarray[Any, np.dtype[np.bool_]] | None:
    """Returns `None` if input is not a DAG

    Since (G_1 @ G_2)[i,j] is nonzero iff there exists k such that G_1[i,k] = G_2[k, j] = True,
    G_1 G_2 represents nodes reachable by using an edge from first G1 and then G2.

    Note that i -> j is in transitive reduction iff there are no other, longer, paths between them.
    Transitive closure is holds paths of _any_ length, thus adj_mat @ tr_closure holds
    all paths of length > 1."""
    tr_closure = transitive_closure(adj_mat)
    if tr_closure is None:
        return None
    return adj_mat & np.logical_not(
        adj_mat.astype(np.intp) @ tr_closure.astype(np.intp)
    )
