# -*- coding: utf-8 -*-
"""
DRUHG: Dialectical Ranking Universal Hierarchical Grouping
Clustering made by self-unrolling the relationships between the objects.
It is most natural clusterization and requires ZERO parameters.
"""

# Author: Pavel "DRUHG" Artamonov
# druhg.p@gmail.com
# License: 3-clause BSD

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClusterMixin
from scipy.sparse import issparse
from sklearn.neighbors import KDTree, BallTree
from sklearn.externals.joblib import Memory
from sklearn.externals import six
from warnings import warn
from sklearn.utils import check_array
from sklearn.externals.joblib.parallel import cpu_count

from ._druhg_tree import UniversalReciprocity
import _druhg_label as labeling

from .plots import MinimumSpanningTree


def druhg(X, max_ranking=16,
          limit1=0, limit2=0, fix_outliers=0,
          metric='minkowski', p=2, leaf_size=40,
          algorithm='best', verbose=False, **kwargs):
    """Perform DRUHG clustering from a vector array or distance matrix.

    Parameters
    ----------
    X : array matrix of shape (n_samples, n_features), or \
            array of shape (n_samples, n_samples)
        A feature array, or array of distances between samples if
        ``metric='precomputed'``.

    max_ranking : int, optional (default=15)
        The maximum number of neighbors to search.
        Affects performance vs precision.

    limit1 : int, optional (default=sqrt(size))
        Clusters that are smaller than this limit treated as noise.
        Use 1 to find True outliers.

    limit2 : int, optional (default=size)
        Clusters with size OVER this limit treated as noise.
        Use it to break down big clusters.

    fix_outliers: int, optional (default=0)
        All outliers will be assigned to the nearest cluster

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
        Exactly, which algorithm to use; DRUHG has variants specialised
        for different characteristics of the data. By default this is set
        to ``best`` which chooses the "best" algorithm given the nature of
        the data. You can force other options if you believe you know
        better. Options are:
            * ``best``
            * ``kdtree``
            * ``balltree``
            * ``none``

    **kwargs : optional
        Arguments passed to the distance metric

    Returns
    -------
    labels : ndarray, shape (n_samples)
        Cluster labels for each point. Noisy samples are given the label -1.

    min_spanning_tree : ndarray, shape (2*n_samples)
        The minimum spanning as edgepairs.

    References
    ----------

    None

    """
    size = X.shape[0]

    if max_ranking is not None:
        if type(max_ranking) is not int:
            raise ValueError('Max ranking must be integer!')
        if max_ranking < 0:
            raise ValueError('Max ranking must be non-negative integer!')

    if leaf_size < 1:
        raise ValueError('Leaf size must be greater than 0!')

    if metric == 'minkowski':
        if p is None:
            raise TypeError('Minkowski metric given but no p value supplied!')
        if p < 0:
            raise ValueError('Minkowski metric with negative p value is not'
                             ' defined!')

    if max_ranking is None:
        max_ranking = 16

    max_ranking = min(size - 1, max_ranking)

    if algorithm == 'best':
        algorithm = 'kd_tree'

    if X.dtype != np.float64:
        print ('Converting to numpy float64')
        X = X.astype(np.float64)

    if "precomputed" in algorithm.lower() or "precomputed" in metric.lower() or issparse(X):
        algorithm = 2
        if issparse(X):
            algorithm = 3
        elif len(X.shape)==2 and X.shape[0] != X.shape[1]:
            raise ValueError('Precomputed matrix is not a square.')
        tree = X
    else:
        # The Cython routines used require contiguous arrays
        if not X.flags['C_CONTIGUOUS']:
            X = np.array(X, dtype=np.double, order='C')

        if "kd" in algorithm.lower() and "tree" in algorithm.lower():
            algorithm = 0
            if metric not in KDTree.valid_metrics:
                raise ValueError('Metric: %s\n'
                                 'Cannot be used with KDTree' % metric)
            tree = KDTree(X, metric=metric, leaf_size=leaf_size, **kwargs)
        elif "ball" in algorithm.lower() and "tree" in algorithm.lower():
            algorithm = 1
            tree = BallTree(X, metric=metric, leaf_size=leaf_size, **kwargs)
        else:
            raise TypeError('Unknown algorithm type %s specified' % algorithm)

    print('algorithm', algorithm, tree)
    ur = UniversalReciprocity(algorithm, tree, max_ranking, metric, leaf_size//3)

    num_edges, pairs = ur.get_tree()
    num_parents, parents = ur.get_clusters_parents()

    labels = labeling.do(size, pairs, num_edges, parents, limit1, limit2, fix_outliers)

    return (labels,
            num_edges, pairs,
            num_parents, parents
            )

class DRUHG(BaseEstimator, ClusterMixin):
    def __init__(self, metric='euclidean',
                 algorithm='best',
                 max_ranking=24,
                 limit1=0,
                 limit2=0,
                 fix_outliers=0,
                 leaf_size=40,
                 verbose=False,
                 **kwargs):
        self.max_ranking = max_ranking
        self.limit1 = limit1
        self.limit2 = limit2
        self.fix_outliers = fix_outliers
        self.metric = metric
        self.algorithm = algorithm
        self.verbose = verbose
        self.leaf_size = leaf_size
        self._metric_kwargs = kwargs

        # self._outlier_scores = None
        # self._prediction_data = None
        self._size = 0
        self._raw_data = None
        self.labels_ = None
        self.mst_ = None
        self.num_edges_ = 0
        self.parents_ = None
        self.num_clusters_ = 0

    def fit(self, X, y=None):
        """Perform DRUHG clustering.

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
        kwargs = self.get_params()
        kwargs.update(self._metric_kwargs)

        self._size = X.shape[0]
        self._raw_data = X

        (self.labels_,
         self.num_edges_,
         self.mst_,
         self.num_clusters_,
         self.parents_) = druhg(X, **kwargs)

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

    def hierarchy(self):
        # converts to standard hierarchical tree format + errors
        # https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

        print ('todo: not done yet')
        return None

    def relabel(self, parents=None, limit1=0, limit2=0, fix_outliers=0):
        """Relabeling with the limit of cluster size.

        Parameters
        ----------

        parents : array of parent-indexes, for surgical removal of certain clusters,
            could be omitted.

        limit1 : clusters under this size are considered as noise.

        limit2 : upper limit for the cluster size,
            resulting clusters would be smaller than this limit.

        fix_outliers : glues outliers to the nearest clusters

        Returns
        -------
        y : ndarray, shape (n_samples, )
            cluster labels,
            -1 are outliers
        """
        if parents == None:
            parents = self.parents_
        return labeling.do(self._size, self.mst_, self.num_edges_, parents, limit1, limit2, fix_outliers)

    @property
    def minimum_spanning_tree_(self):
        if self.mst_ is not None:
            if self._raw_data is not None:
                return MinimumSpanningTree(self.mst_,
                                           self._raw_data,
                                           self.labels_)
            else:
                warn('No raw data is available.')
                return None
        else:
            raise AttributeError('No minimum spanning tree was generated.')
