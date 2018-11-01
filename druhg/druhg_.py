# -*- coding: utf-8 -*-
"""
DRUHG: Density Ranking Universal Hierarchical Grouping
metric space with even subjective ranking distances
"""

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import pairwise_distances
from scipy.sparse import issparse
from sklearn.neighbors import KDTree, BallTree
from sklearn.externals.joblib import Memory
from sklearn.externals import six
from warnings import warn
from sklearn.utils import check_array
from sklearn.externals.joblib.parallel import cpu_count

from scipy.sparse import csgraph

from ._druhg_helper import (make_hierarchy)
from ._hdbscan_tree import (condense_tree,
                            compute_stability,
                            get_clusters,
                            outlier_scores)

from ._druhg_even_rankability import (even_rankability, mst_linkage_core)  # , sparse_mutual_rankability)

from ._druhg_boruvka import KDTreeBoruvkaAlgorithm, BallTreeBoruvkaAlgorithm
from ._druhg_prims import MSTPrimsAlgorithm

from .dist_metrics import DistanceMetric

from .plots import CondensedTree, SingleLinkageTree, MinimumSpanningTree

FAST_METRICS = (KDTree.valid_metrics + BallTree.valid_metrics +
                ['cosine', 'arccos'])

# Author: Leland McInnes <leland.mcinnes@gmail.com>
#         Steve Astels <sastels@gmail.com>
#         John Healy <jchealy@gmail.com>
#         Pavel "DRUHG" Artamonov <main.edgehog.net@gmail.com>
# License: BSD 3 clause
from numpy import isclose

def _tree_to_labels(single_linkage_tree,
                    min_samples,
                    cluster_selection_method='eom',
                    allow_single_cluster=False,
                    match_reference_implementation=False):
    """Converts a pretrained tree and cluster size into a
    set of labels and probabilities.
    """
    print ('min_samples for condensed tree', min_samples)

    condensed_tree = condense_tree(single_linkage_tree, min_samples)

    stability_dict = compute_stability(condensed_tree)
    labels, probabilities, stabilities = get_clusters(condensed_tree,
                                                      stability_dict,
                                                      cluster_selection_method,
                                                      allow_single_cluster,
                                                      match_reference_implementation)

    return (labels, probabilities, stabilities, condensed_tree,
            single_linkage_tree)

def _druhg_none(X, alpha=1.0, metric='minkowski', p=2,
                max_ranking=0, min_ranking=1, run_times=1,
                leaf_size=None, gen_min_span_tree=False, **kwargs):
    # for mst building
    if metric == 'minkowski':
        distance_matrix = pairwise_distances(X, metric=metric, p=p)
    elif metric == 'arccos':
        distance_matrix = pairwise_distances(X, metric='cosine', **kwargs)
    elif metric == 'precomputed':
        # Treating this case explicitly, instead of letting
        #   sklearn.metrics.pairwise_distances handle it,
        #   enables the usage of numpy.inf in the distance
        #   matrix to indicate missing distance information.
        # TODO: Check if copying is necessary
        distance_matrix = X.copy()
    else:
        distance_matrix = pairwise_distances(X, metric=metric, **kwargs)

    # if issparse(distance_matrix):
    #     # raise TypeError('Sparse distance matrices not yet supported')
    #     return _druhg_sparse_distance_matrix(distance_matrix, min_samples,
    #                                            alpha, metric, p,
    #                                            leaf_size, gen_min_span_tree,
    #                                            **kwargs)
    size = distance_matrix.shape[0]

    print('none-' + metric, '  size: ', str(size), ' edges: ', str(
        size * (size - 1) / 2))  # , 'diff_edges: ', str(len(np.unique(distance_matrix)))

    min_spanning_tree = mst_linkage_core(distance_matrix)

    # Warn if the MST couldn't be constructed around the missing distances
    if np.isinf(min_spanning_tree.T[2]).any():
        warn('The minimum spanning tree contains edge weights with value '
             'infinity. Potentially, you are missing too many distances '
             'in the initial distance matrix for the given neighborhood '
             'size.', UserWarning)

    # mst_linkage_core does not generate a full minimal spanning tree
    # If a tree is required then we must build the edges from the information
    # returned by mst_linkage_core (i.e. just the order of points to be merged)
    if gen_min_span_tree:
        result_min_span_tree = min_spanning_tree.copy()
        for index, row in enumerate(result_min_span_tree[1:], 1):
            candidates = np.where(isclose(distance_matrix[int(row[1])],
                                          row[2]))[0]
            candidates = np.intersect1d(candidates,
                                        min_spanning_tree[:index, :2].astype(
                                            int))
            candidates = candidates[candidates != row[1]]
            assert len(candidates) > 0
            row[0] = candidates[0]
    else:
        result_min_span_tree = None

    # Sort edges of the min_spanning_tree by weight
    min_spanning_tree = min_spanning_tree[np.argsort(min_spanning_tree.T[2]),
                        :]

    # Convert edge list into standard hierarchical clustering format
    single_linkage_tree = make_hierarchy(min_spanning_tree)

    return single_linkage_tree, result_min_span_tree


def _druhg_generic(X, alpha=1.0, metric='minkowski', p=2,
                   max_ranking=0, min_ranking=1, run_times=1,
                   leaf_size=None, gen_min_span_tree=False, **kwargs):
    if metric == 'minkowski':
        distance_matrix = pairwise_distances(X, metric=metric, p=p)
    elif metric == 'arccos':
        distance_matrix = pairwise_distances(X, metric='cosine', **kwargs)
    elif metric == 'precomputed':
        # Treating this case explicitly, instead of letting
        #   sklearn.metrics.pairwise_distances handle it,
        #   enables the usage of numpy.inf in the distance
        #   matrix to indicate missing distance information.
        # TODO: Check if copying is necessary
        distance_matrix = X.copy()
    else:
        distance_matrix = pairwise_distances(X, metric=metric, **kwargs)

    # if issparse(distance_matrix):
    #     # raise TypeError('Sparse distance matrices not yet supported')
    #     return _druhg_sparse_distance_matrix(distance_matrix, min_samples,
    #                                            alpha, metric, p,
    #                                            leaf_size, gen_min_span_tree,
    #                                            **kwargs)
    size = distance_matrix.shape[0]

    print ('generic-' + metric + '. min_ranking', min_ranking, 'max_ranking', max_ranking, '  size: ', str(size), ' edges: ', str(
        size * (size - 1) / 2))  # , 'diff_edges: ', str(len(np.unique(distance_matrix)))

    even_rankability_ = even_rankability(distance_matrix, min_flatting=min_ranking, max_neighbors_search=max_ranking)

    print ('run:', str(1))  # , 'diff_edges: ', str(len(np.unique(even_rankability_)))

    run_times -= 1
    i = 1
    while (run_times > 0):
        even_rankability_ = even_rankability(even_rankability_, min_flatting=min_ranking)
        run_times -= 1
        print ('run:', str(i))  # , 'diff_edges: ', str(len(np.unique(even_rankability_)))
        i += 1
        # print even_rankability_.round(2)


    min_spanning_tree = mst_linkage_core(even_rankability_)

    # Warn if the MST couldn't be constructed around the missing distances
    if np.isinf(min_spanning_tree.T[2]).any():
        warn('The minimum spanning tree contains edge weights with value '
             'infinity. Potentially, you are missing too many distances '
             'in the initial distance matrix for the given neighborhood '
             'size.', UserWarning)

    # mst_linkage_core does not generate a full minimal spanning tree
    # If a tree is required then we must build the edges from the information
    # returned by mst_linkage_core (i.e. just the order of points to be merged)
    if gen_min_span_tree:
        result_min_span_tree = min_spanning_tree.copy()
        for index, row in enumerate(result_min_span_tree[1:], 1):
            candidates = np.where(isclose(even_rankability_[int(row[1])],
                                          row[2]))[0]
            candidates = np.intersect1d(candidates,
                                        min_spanning_tree[:index, :2].astype(
                                            int))
            candidates = candidates[candidates != row[1]]
            assert len(candidates) > 0
            row[0] = candidates[0]
    else:
        result_min_span_tree = None

    # Sort edges of the min_spanning_tree by weight
    min_spanning_tree = min_spanning_tree[np.argsort(min_spanning_tree.T[2]),
                        :]

    # Convert edge list into standard hierarchical clustering format
    single_linkage_tree = make_hierarchy(min_spanning_tree)

    return single_linkage_tree, result_min_span_tree


#
# def _druhg_sparse_distance_matrix(X, min_samples=5, alpha=1.0,
#                                     metric='minkowski', p=2, leaf_size=40,
#                                     gen_min_span_tree=False, **kwargs):
#     assert issparse(X)
#
#     lil_matrix = X.tolil()
#
#     # Compute sparse mutual rankability graph
#     mutual_rankability_ = sparse_mutual_rankability(lil_matrix,
#                                                       min_points=min_samples)
#
#     if csgraph.connected_components(mutual_rankability_, directed=False,
#                                     return_labels=False) > 1:
#         raise ValueError('Sparse distance matrix has multiple connected'
#                          ' components!\nThat is, there exist groups of points '
#                          'that are completely disjoint -- there are no distance '
#                          'relations connecting them\n'
#                          'Run DRUHG on each component.')
#
#     # Compute the minimum spanning tree for the sparse graph
#     sparse_min_spanning_tree = csgraph.minimum_spanning_tree(
#         mutual_rankability_)
#
#     # Convert the graph to scipy cluster array format
#     nonzeros = sparse_min_spanning_tree.nonzero()
#     nonzero_vals = sparse_min_spanning_tree[nonzeros]
#     min_spanning_tree = np.vstack(nonzeros + (nonzero_vals,)).T
#
#     # Sort edges of the min_spanning_tree by weight
#     min_spanning_tree = min_spanning_tree[np.argsort(min_spanning_tree.T[2]),
#                         :][0]
#
#     # Convert edge list into standard hierarchical clustering format
#     single_linkage_tree = label(min_spanning_tree)
#
#     if gen_min_span_tree:
#         return single_linkage_tree, min_spanning_tree
#     else:
#         return single_linkage_tree, None
#

def _druhg_boruvka_kdtree(X, max_ranking=16, min_ranking=1, alpha=1.0,
                          metric='minkowski', p=2, leaf_size=40,
                          approx_min_span_tree=True,
                          gen_min_span_tree=False,
                          core_dist_n_jobs=4, **kwargs):
    if leaf_size < 3:
        leaf_size = 3

    if core_dist_n_jobs < 1:
        core_dist_n_jobs = max(cpu_count() + 1 + core_dist_n_jobs, 1)

    if X.dtype != np.float64:
        X = X.astype(np.float64)

    print ('boruvka_kdtree-' + metric + ' size:', str(len(X)), ' edges:', str(
        len(X) * (len(
            X) - 1) / 2), ' max_ranking:', max_ranking, ' min_ranking:', min_ranking)  # , 'diff_edges: ', str(len(np.unique(distance_matrix)))

    tree = KDTree(X, metric=metric, leaf_size=leaf_size, **kwargs)
    # todo: count pushed edges
    alg = KDTreeBoruvkaAlgorithm(tree, max_neighbors_search=max_ranking, min_flatting=min_ranking, metric=metric,
                                 leaf_size=leaf_size // 3,
                                 approx_min_span_tree=approx_min_span_tree,
                                 n_jobs=core_dist_n_jobs, **kwargs)
    min_spanning_tree = alg.spanning_tree()
    # Sort edges of the min_spanning_tree by weight
    row_order = np.argsort(min_spanning_tree.T[2])
    min_spanning_tree = min_spanning_tree[row_order, :]
    # Convert edge list into standard hierarchical clustering format
    single_linkage_tree = make_hierarchy(min_spanning_tree)

    if gen_min_span_tree:
        return single_linkage_tree, min_spanning_tree
    else:
        return single_linkage_tree, None


def _druhg_boruvka_balltree(X, max_ranking=16, min_ranking=1, alpha=1.0,
                            metric='minkowski', p=2, leaf_size=40,
                            approx_min_span_tree=True,
                            gen_min_span_tree=False,
                            core_dist_n_jobs=4, **kwargs):
    if leaf_size < 3:
        leaf_size = 3

    if core_dist_n_jobs < 1:
        core_dist_n_jobs = max(cpu_count() + 1 + core_dist_n_jobs, 1)

    if X.dtype != np.float64:
        X = X.astype(np.float64)

    print ('boruvka_balltree-' + metric + ' size:', str(len(X)), ' edges:', str(
        len(X) * (len(
            X) - 1) / 2), ' max_ranking:', max_ranking, ' min_ranking:', min_ranking)  # , 'diff_edges: ', str(len(np.unique(distance_matrix)))

    tree = BallTree(X, metric=metric, leaf_size=leaf_size, **kwargs)
    # todo: count pushed edges
    alg = BallTreeBoruvkaAlgorithm(tree, max_neighbors_search=max_ranking, min_flatting=min_ranking, metric=metric,
                                   leaf_size=leaf_size // 3,
                                   approx_min_span_tree=approx_min_span_tree,
                                   n_jobs=core_dist_n_jobs, **kwargs)
    min_spanning_tree = alg.spanning_tree()
    # Sort edges of the min_spanning_tree by weight
    min_spanning_tree = min_spanning_tree[np.argsort(min_spanning_tree.T[2]),
                        :]
    # Convert edge list into standard hierarchical clustering format
    single_linkage_tree = make_hierarchy(min_spanning_tree)

    if gen_min_span_tree:
        return single_linkage_tree, min_spanning_tree
    else:
        return single_linkage_tree, None


def _druhg_prims_kdtree(X, max_ranking=16, min_ranking=1, alpha=1.0,
                        metric='minkowski', p=2, leaf_size=40,
                        gen_min_span_tree=False, **kwargs):
    if X.dtype != np.float64:
        X = X.astype(np.float64)

    # The Cython routines used require contiguous arrays
    if not X.flags['C_CONTIGUOUS']:
        X = np.array(X, dtype=np.double, order='C')

    print ('prims_kdtree-' + metric + ' size:', str(len(X)), ' edges:', str(
        len(X) * (len(
            X) - 1) / 2), ' max_ranking:', max_ranking, ' min_ranking:', min_ranking)  # , 'diff_edges: ', str(len(np.unique(distance_matrix)))

    tree = KDTree(X, metric=metric, leaf_size=leaf_size, **kwargs)
    # todo: count pushed edges
    # TODO: Deal with p for minkowski appropriately
    alg = MSTPrimsAlgorithm(tree, is_kd_tree=1,
                            max_neighbors_search=max_ranking, min_flatting=min_ranking,
                            metric=metric,
                            leaf_size=leaf_size // 3,
                            alpha=1.0,
                            **kwargs)
    min_spanning_tree = alg.spanning_tree()

    # Sort edges of the min_spanning_tree by weight
    row_order = np.argsort(min_spanning_tree.T[2])
    min_spanning_tree = min_spanning_tree[row_order, :]

    # Convert edge list into standard hierarchical clustering format
    single_linkage_tree = make_hierarchy(min_spanning_tree)

    return single_linkage_tree, None


def _druhg_prims_balltree(X, max_ranking=16, min_ranking=1, alpha=1.0,
                          metric='minkowski', p=2, leaf_size=40,
                          gen_min_span_tree=False, **kwargs):
    if X.dtype != np.float64:
        X = X.astype(np.float64)

    # The Cython routines used require contiguous arrays
    if not X.flags['C_CONTIGUOUS']:
        X = np.array(X, dtype=np.double, order='C')

    print ('prims_balltree-' + metric + ' size:', str(len(X)), ' edges:', str(
        len(X) * (len(
            X) - 1) / 2), ' max_ranking:', max_ranking, ' min_ranking:', min_ranking)  # , 'diff_edges: ', str(len(np.unique(distance_matrix)))

    tree = BallTree(X, metric=metric, leaf_size=leaf_size, **kwargs)
    # todo: count pushed edges
    # TODO: Deal with p for minkowski appropriately
    alg = MSTPrimsAlgorithm(tree, is_kd_tree=0,
                            max_neighbors_search=max_ranking, min_flatting=min_ranking,
                            metric=metric,
                            leaf_size=leaf_size // 3,
                            alpha=1.0,
                            **kwargs)
    min_spanning_tree = alg.spanning_tree()

    # Sort edges of the min_spanning_tree by weight
    row_order = np.argsort(min_spanning_tree.T[2])
    min_spanning_tree = min_spanning_tree[row_order, :]

    # Convert edge list into standard hierarchical clustering format
    single_linkage_tree = make_hierarchy(min_spanning_tree)

    return single_linkage_tree, None


def check_precomputed_distance_matrix(X):
    """Perform check_array(X) after removing infinite values (numpy.inf) from the given distance matrix.
    """
    tmp = X.copy()
    tmp[np.isinf(tmp)] = 1
    check_array(tmp)


def druhg(X, max_ranking=16, min_ranking=1, min_samples=5, alpha=1.0,
          metric='minkowski', p=2, run_times=0, leaf_size=40,
          algorithm='best', memory=Memory(cachedir=None, verbose=0),
          approx_min_span_tree=True, gen_min_span_tree=False,
          core_dist_n_jobs=4,
          cluster_selection_method='eom', allow_single_cluster=False,
          match_reference_implementation=False, **kwargs):
    """Perform DRUHG clustering from a vector array or distance matrix.

    Parameters
    ----------
    X : array matrix of shape (n_samples, n_features), or \
            array of shape (n_samples, n_samples)
        A feature array, or array of distances between samples if
        ``metric='precomputed'``.
    min_ranking : int, optional (default=1)
        The minimum ranking to use in even ranking distance.
        Clusters with density less than that will be merged with clusters up to this ranking

    max_ranking : int, optional (default=None)
        The maximum number of neighbors to search.
        Use it as an upper bound of cluster-density as a performance boost

    min_samples : int, optional (default=5)
        The number of samples in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
        For final interpretation only and rerun-revisualizes.

    alpha : float, optional (default=1.0)
        A distance scaling parameter as used in robust single linkage.
        See [2]_ for more information.

    metric : string or callable, optional (default='minkowski')
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by metrics.pairwise.pairwise_distances for its
        metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square.

    p : int, optional (default=2)
        p value to use if using the minkowski metric.

    leaf_size : int, optional (default=40)
        Leaf size for trees responsible for fast nearest
        neighbour queries.

    algorithm : string, optional (default='best')
        Exactly which algorithm to use; DRUHG has variants specialised
        for different characteristics of the data. By default this is set
        to ``best`` which chooses the "best" algorithm given the nature of
        the data. You can force other options if you believe you know
        better. Options are:
            * ``best``
            * ``generic``
            * ``boruvka_kdtree``
            * ``boruvka_balltree``
            * ``prims_kdtree``
            * ``prims_balltree``
            * ``none``

    memory : instance of joblib.Memory or string, optional
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    approx_min_span_tree : bool, optional (default=True)
        Whether to accept an only approximate minimum spanning tree.
        For some algorithms this can provide a significant speedup, but
        the resulting clustering may be of marginally lower quality.
        If you are willing to sacrifice speed for correctness you may want
        to explore this; in general this should be left at the default True.

    gen_min_span_tree : bool, optional (default=False)
        Whether to generate the minimum spanning tree for later analysis.

    core_dist_n_jobs : int, optional (default=4)
        Number of parallel jobs to run in core distance computations (if
        supported by the specific algorithm). For ``core_dist_n_jobs``
        below -1, (n_cpus + 1 + core_dist_n_jobs) are used.

    cluster_selection_method : string, optional (default='eom')
        The method used to select clusters from the condensed tree. The
        standard approach for DRUHG* is to use an Excess of Mass algorithm
        to find the most persistent clusters. Alternatively you can instead
        select the clusters at the leaves of the tree -- this provides the
        most fine grained and homogeneous clusters. Options are:
            * ``eom``
            * ``leaf``

    allow_single_cluster : bool, optional (default=False)
        By default DRUHG* will not produce a single cluster, setting this
        to t=True will override this and allow single cluster results in
        the case that you feel this is a valid result for your dataset.
        (default False)


    **kwargs : optional
        Arguments passed to the distance metric

    Returns
    -------
    labels : ndarray, shape (n_samples, )
        Cluster labels for each point.  Noisy samples are given the label -1.

    probabilities : ndarray, shape (n_samples, )
        Cluster membership strengths for each point. Noisy samples are assigned
        0.

    cluster_persistence : array, shape  (n_clusters, )
        A score of how persistent each cluster is. A score of 1.0 represents
        a perfectly stable cluster that persists over all distance scales,
        while a score of 0.0 represents a perfectly ephemeral cluster. These
        scores can be guage the relative coherence of the clusters output
        by the algorithm.

    condensed_tree : record array
        The condensed cluster hierarchy used to generate clusters.

    single_linkage_tree : ndarray, shape (n_samples - 1, 4)
        The single linkage tree produced during clustering in scipy
        hierarchical clustering format
        (see http://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html).

    min_spanning_tree : ndarray, shape (n_samples - 1, 3)
        The minimum spanning as an edgelist. If gen_min_span_tree was False
        this will be None.

    References
    ----------

    .. [1] Campello, R. J., Moulavi, D., & Sander, J. (2013, April).
       Density-based clustering based on hierarchical density estimates.
       In Pacific-Asia Conference on Knowledge Discovery and Data Mining
       (pp. 160-172). Springer Berlin Heidelberg.

    .. [2] Chaudhuri, K., & Dasgupta, S. (2010). Rates of convergence for the
       cluster tree. In Advances in Neural Information Processing Systems
       (pp. 343-351).

    """
    if min_samples is None:
        min_samples = 2

    if type(min_samples) is not int or type(min_ranking) is not int:
        raise ValueError('Min samples and min ranking must be integers!')

    if max_ranking is not None and type(max_ranking) is not int:
        raise ValueError('Max ranking must be integer!')

    if max_ranking is not None and max_ranking < 0:
        raise ValueError('Max ranking must be non-negative integer!')

    if min_samples <= 0:
        raise ValueError('Min samples must be positive integer')

    if min_ranking == 0:
        min_ranking = 1
    if min_ranking < 0:
        raise ValueError('Min ranking must be positive integer')

    if not isinstance(alpha, float) or alpha <= 0.0:
        raise ValueError('Alpha must be a positive float value greater than'
                         ' 0!')

    if leaf_size < 1:
        raise ValueError('Leaf size must be greater than 0!')

    if metric == 'minkowski':
        if p is None:
            raise TypeError('Minkowski metric given but no p value supplied!')
        if p < 0:
            raise ValueError('Minkowski metric with negative p value is not'
                             ' defined!')

    if cluster_selection_method not in ('eom', 'leaf'):
        raise ValueError('Invalid Cluster Selection Method: %s\n'
                         'Should be one of: "eom", "leaf"\n')

    # Checks input and converts to an nd-array where possible
    if metric != 'precomputed' or issparse(X):
        X = check_array(X, accept_sparse='csr')
    else:
        # Only non-sparse, precomputed distance matrices are handled here
        #   and thereby allowed to contain numpy.inf for missing distances
        check_precomputed_distance_matrix(X)

    # Python 2 and 3 compliant string_type checking
    if isinstance(memory, six.string_types):
        memory = Memory(cachedir=memory, verbose=0)

    size = X.shape[0]
    min_samples = min(size, min_samples)
    if min_samples <= 0:
        min_samples = 2

    max_ranking = min(size - 1, max_ranking)

    if algorithm == 'best':
        if metric != "precomputed" and metric not in FAST_METRICS and size > 1000:
            if metric in KDTree.valid_metrics:
                if X.shape[1] <= 40:
                    algorithm = 'boruvka_kdtree'
                else:
                    algorithm = 'prims_kdtree'
            else:
                if X.shape[1] <= 40:
                    algorithm = 'boruvka_balltree'
                else:
                    algorithm = 'prims_balltree'
        else:
            algorithm = 'generic'
        print ('best algorithm chosen:  ' + str(algorithm) + '-' + str(metric))

    if algorithm != 'generic' and max_ranking is None:
        max_ranking = 16

    if algorithm == 'none':
        (single_linkage_tree,
         result_min_span_tree) = memory.cache(
            _druhg_none)(X, alpha, metric,
                            p, max_ranking, min_ranking, run_times, leaf_size, gen_min_span_tree, **kwargs)
    elif algorithm == 'generic':
        if max_ranking is None:
            max_ranking = size - 1
        (single_linkage_tree,
         result_min_span_tree) = memory.cache(
            _druhg_generic)(X, alpha, metric,
                            p, max_ranking, min_ranking, run_times, leaf_size, gen_min_span_tree, **kwargs)
    elif algorithm == 'boruvka_kdtree':
        if metric not in BallTree.valid_metrics:
            raise ValueError("Cannot use Boruvka with KDTree for this"
                             " metric!")
        (single_linkage_tree, result_min_span_tree) = memory.cache(
            _druhg_boruvka_kdtree)(X, max_ranking, min_ranking, alpha,
                                   metric, p, leaf_size,
                                   approx_min_span_tree,
                                   gen_min_span_tree,
                                   core_dist_n_jobs, **kwargs)
    elif algorithm == 'boruvka_balltree':
        if metric not in BallTree.valid_metrics:
            raise ValueError("Cannot use Boruvka with BallTree for this"
                             " metric!")
        (single_linkage_tree, result_min_span_tree) = memory.cache(
            _druhg_boruvka_balltree)(X, max_ranking, min_ranking, alpha,
                                     metric, p, leaf_size,
                                     approx_min_span_tree,
                                     gen_min_span_tree,
                                     core_dist_n_jobs, **kwargs)
    elif algorithm == 'prims_kdtree':
        if metric not in KDTree.valid_metrics:
            raise ValueError("Cannot use Prim's with KDTree for this"
                             " metric!")
        (single_linkage_tree, result_min_span_tree) = memory.cache(
            _druhg_prims_kdtree)(X, max_ranking, min_ranking, alpha,
                                 metric, p, leaf_size,
                                 gen_min_span_tree, **kwargs)
    elif algorithm == 'prims_balltree':
        if metric not in BallTree.valid_metrics:
            raise ValueError("Cannot use Prim's with BallTree for this"
                             " metric!")
        (single_linkage_tree, result_min_span_tree) = memory.cache(
            _druhg_prims_balltree)(X, max_ranking, min_ranking, alpha,
                                   metric, p, leaf_size,
                                   gen_min_span_tree, **kwargs)
    else:
        raise TypeError('Unknown algorithm type %s specified' % algorithm)

    return _tree_to_labels(single_linkage_tree,
                           min_samples,
                           cluster_selection_method,
                           allow_single_cluster,
                           match_reference_implementation) + \
           (result_min_span_tree,)


class DRUHG(BaseEstimator, ClusterMixin):
    def __init__(self, max_ranking=16, min_ranking=1, min_samples=5,
                 metric='euclidean', alpha=1.0, p=None,
                 algorithm='best', run_times=0, leaf_size=40,
                 memory=Memory(cachedir=None, verbose=0),
                 approx_min_span_tree=True,
                 gen_min_span_tree=False,
                 core_dist_n_jobs=4,
                 cluster_selection_method='eom',
                 allow_single_cluster=False,
                 prediction_data=False,
                 match_reference_implementation=False, **kwargs):
        self.max_ranking = max_ranking
        self.min_ranking = min_ranking
        self.min_samples = min_samples
        self.alpha = alpha
        self.run_times = run_times
        self.metric = metric
        self.p = p
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.memory = memory
        self.approx_min_span_tree = approx_min_span_tree
        self.gen_min_span_tree = gen_min_span_tree
        self.core_dist_n_jobs = core_dist_n_jobs
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.match_reference_implementation = match_reference_implementation
        self.prediction_data = prediction_data
        self._metric_kwargs = kwargs
        self._condensed_tree = None
        self._single_linkage_tree = None
        self._min_spanning_tree = None
        self._raw_data = None
        self._outlier_scores = None
        self._prediction_data = None

    def fit(self, X, y=None):
        """Perform DRUHG clustering from features or distance matrix.

        Parameters
        ----------
        X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
                array of shape (n_samples, n_samples)
            A feature array, or array of distances between samples if
            ``metric='precomputed'``.

        Returns
        -------
        self : object
            Returns self
        """
        if self.metric != 'precomputed':
            X = check_array(X, accept_sparse='csr')
            self._raw_data = X
        elif issparse(X):
            # Handle sparse precomputed distance matrices separately
            X = check_array(X, accept_sparse='csr')
        else:
            # Only non-sparse, precomputed distance matrices are allowed
            #   to have numpy.inf values indicating missing distances
            check_precomputed_distance_matrix(X)

        kwargs = self.get_params()
        # prediction data only applies to the persistent model, so remove
        # it from the keyword args we pass on the the function
        kwargs.pop('prediction_data', None)
        kwargs.update(self._metric_kwargs)

        (self.labels_,
         self.probabilities_,
         self.cluster_persistence_,
         self._condensed_tree,
         self._single_linkage_tree,
         self._min_spanning_tree) = druhg(X, **kwargs)

        if self.prediction_data:
            self.generate_prediction_data()

        return self

    def fit_predict(self, X, y=None):
        """Performs clustering on X and returns cluster labels.

        Parameters
        ----------
        X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
                array of shape (n_samples, n_samples)
            A feature array, or array of distances between samples if
            ``metric='precomputed'``.

        Returns
        -------
        y : ndarray, shape (n_samples, )
            cluster labels
        """
        self.fit(X)
        return self.labels_

    def revisualize(self, min_samples):
        self.min_samples = min_samples

        (self.labels_,
         self.probabilities_,
         self.cluster_persistence_,
         self._condensed_tree,
         self._single_linkage_tree) = _tree_to_labels(self._single_linkage_tree,
                                                      self.min_samples,
                                                      self.cluster_selection_method,
                                                      self.allow_single_cluster,
                                                      self.match_reference_implementation)
        return self
        # return (labels, probabilities, stabilities,  condensed_tree,
        #         single_linkage_tree)

        # return self.labels_

    def generate_prediction_data(self):
        """
        Create data that caches intermediate results used for predicting
        the label of new/unseen points. This data is only useful if
        you are intending to use functions from ``DRUHG.prediction``.
        """

        if self.metric in FAST_METRICS:
            min_samples = self.min_samples or self.max_ranking
            if self.metric in KDTree.valid_metrics:
                tree_type = 'kdtree'
            elif self.metric in BallTree.valid_metrics:
                tree_type = 'balltree'
            else:
                warn('Metric {} not supported for prediction data!'.format(self.metric))
                return

            self._prediction_data = PredictionData(
                self._raw_data, self.condensed_tree_, min_samples,
                tree_type=tree_type, metric=self.metric,
                **self._metric_kwargs
            )
        else:
            warn('Cannot generate prediction data for non-vector'
                 'space inputs -- access to the source data rather'
                 'than mere distances is required!')

    @property
    def prediction_data_(self):
        if self._prediction_data is None:
            raise AttributeError('No prediction data was generated')
        else:
            return self._prediction_data

    @property
    def outlier_scores_(self):
        if self._outlier_scores is not None:
            return self._outlier_scores
        else:
            if self._condensed_tree is not None:
                self._outlier_scores = outlier_scores(self._condensed_tree)
                return self._outlier_scores
            else:
                raise AttributeError('No condensed tree was generated; try running fit first.')

    @property
    def condensed_tree_(self):
        if self._condensed_tree is not None:
            return CondensedTree(self._condensed_tree,
                                 self.cluster_selection_method,
                                 self.allow_single_cluster)
        else:
            raise AttributeError('No condensed tree was generated; try running fit first.')

    @property
    def single_linkage_tree_(self):
        if self._single_linkage_tree is not None:
            return SingleLinkageTree(self._single_linkage_tree)
        else:
            raise AttributeError('No single linkage tree was generated; try running fit'
                                 ' first.')

    @property
    def minimum_spanning_tree_(self):
        if self._min_spanning_tree is not None:
            if self._raw_data is not None:
                return MinimumSpanningTree(self._min_spanning_tree,
                                           self._raw_data)
            else:
                warn('No raw data is available; this may be due to using'
                     ' a precomputed metric matrix. No minimum spanning'
                     ' tree will be provided without raw data.')
                return None
        else:
            raise AttributeError('No minimum spanning tree was generated.'
                                 'This may be due to optimized algorithm variations that skip'
                                 ' explicit generation of the spanning tree.')

    @property
    def exemplars_(self):
        if self._prediction_data is not None:
            return self._prediction_data.exemplars
        elif self.metric in FAST_METRICS:
            self.generate_prediction_data()
            return self._prediction_data.exemplars
        else:
            raise AttributeError('Currently exemplars require the use of vector input data'
                                 'with a suitable metric. This will likely change in the '
                                 'future, but for now no exemplars can be provided')
