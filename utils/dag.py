"""Directed Acyclic Graph (DAG) utilities

Notes on notation.
Let G be a graph on n nodes with adjacency matrix A. Then, A[i, j] = True <=> i -> j in G.
This means that parents of node i in G are exactly non-zero entries in i-th column of A.
"""
from typing import Any

import numpy as np
import numpy.typing as npt
import networkx as nx
import itertools

def graph_diff(
    g1: npt.NDArray[np.bool_],
    g2: npt.NDArray[np.bool_],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    edges1 = set(zip(np.where(g1)[0], np.where(g1)[1]))
    edges2 = set(zip(np.where(g2)[0], np.where(g2)[1]))

    g1_reversed = {(j, i) for (i, j) in edges1 if i != j}
    g2_reversed = {(j, i) for (i, j) in edges2 if i != j}

    additions = edges2 - edges1 - g1_reversed
    deletions = edges1 - edges2 - g2_reversed
    reversals = edges1 & g2_reversed

    return additions, deletions, reversals


def cm_graph_entries(
    g1: npt.NDArray[np.bool_],
    g2: npt.NDArray[np.bool_],
) -> tuple[int, int, int]:
    """Computes TP, FP, FN, TN for the edges of
    g1: true graph
    g2: estimated graph
    """
    # g1: true graph
    # g2: estimated graph
    tp =  g1 &  g2
    fn =  g1 & ~g2
    fp = ~g1 &  g2
    tn = ~g1 & ~g2
    return tp.sum(dtype=int), fp.sum(dtype=int), fn.sum(dtype=int), tn.sum(dtype=int)

def precision_recall_f1_graph(g1: npt.NDArray[np.bool_], g2: npt.NDArray[np.bool_]) -> tuple[float, float, float]:
    tp, fp, fn, _ = cm_graph_entries(g1, g2)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1

def structural_hamming_distance(g1: np.ndarray, g2: np.ndarray) -> int:
    additions, deletions, reversals = graph_diff(g1, g2)
    return len(additions) + len(deletions) + len(reversals)


def find_all_top_order(adj_mat: npt.NDArray[np.bool_]) -> list[npt.NDArray[np.intp]]:
    g = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph)
    return list(map(lambda l: np.array(l, dtype=np.intp), nx.all_topological_sorts(g)))


def topological_order(adj_mat: npt.NDArray[np.bool_]) -> npt.NDArray[np.intp] | None:
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


def surrounding_mat(adj_mat: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_] | None:
    (n, _) = adj_mat.shape
    sur_mat = np.zeros_like(adj_mat)
    top_order = topological_order(adj_mat)
    if top_order is None:
        return None
    for i in top_order:
        pa_i = adj_mat[:, i].nonzero()[0]
        ch_i = adj_mat[i, :].nonzero()[0]
        for j in pa_i:
            ch_j = adj_mat[j, :].nonzero()[0]
            if np.all(np.isin(ch_i, ch_j)):
                sur_mat[j, i] = True

    return sur_mat

def transitive_closure(adj_mat: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_] | None:
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


def transitive_reduction(adj_mat: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_] | None:
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

def confusion_mat_graph(true_g: npt.NDArray[np.bool_], hat_g: npt.NDArray[np.bool_]) -> list[list[int]]:
    edge_cm = [
        [
            ( true_g &  hat_g).sum(dtype=int),
            (~true_g &  hat_g).sum(dtype=int),
        ], [
            ( true_g & ~hat_g).sum(dtype=int),
            (~true_g & ~hat_g).sum(dtype=int),
        ]
    ]
    return edge_cm


def closest_dag(g_hat: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """
    Return the closest DAG to the input graph (by maximizing the number of edges
    under a permutation that makes the graph upper-triangular).

    Args:
        g_hat: Input graph, shape (n, n), dtype bool.

    Returns:
        A graph of shape (n, n), dtype bool, that is a DAG.
    """
    # Number of nodes
    n = g_hat.shape[0]

    # Initialize the best permutation and best edge count
    best_perm = np.arange(n)
    best_edges = 0

    # Try all permutations of [0, 1, 2, ..., n-1]
    for perm_tuple in itertools.permutations(range(n)):
        perm = np.array(perm_tuple, dtype=int)
        # Count edges in the upper-triangular part after reordering by perm
        reordered = g_hat[np.ix_(perm, perm)]
        edges = np.sum(np.triu(reordered, k=0))
        if edges > best_edges:
            best_edges = edges
            best_perm = perm

    # Invert the best permutation
    best_perm_inv = np.argsort(best_perm)

    # Reorder g_hat by the best permutation
    g_reordered = g_hat[np.ix_(best_perm, best_perm)]
    # Keep only the strictly upper-triangular edges to ensure acyclicity
    g_reordered = np.triu(g_reordered, k=1)
    # Revert to the original node ordering
    g_reordered = g_reordered[np.ix_(best_perm_inv, best_perm_inv)]

    return g_reordered

