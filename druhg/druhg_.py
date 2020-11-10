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
          limit1=None, limit2=None, exclude = None, fix_outliers=0,
          metric='minkowski', p=2,
          algorithm='best', leaf_size=40,
          verbose=False, **kwargs):
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

    limit2 : int, optional (default=size/2)
        Clusters with size OVER this limit treated as noise.
        Use it to break down big clusters.

    exclude: list, optional (default=None)
        Clusters with these indexes would not be formed.
        Use it for surgical cluster removal.

    fix_outliers: int, optional (default=0)
        In case of 1 - all outliers will be assigned to the nearest cluster

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
        Exactly, which algorithm to use; DRUHG has variants specialized
        for different characteristics of the data. By default, this is set
        to ``best`` which chooses the "best" algorithm given the nature of
        the data. You can force other options if you believe you know
        better. Options are:
            * ``best``
            * ``kdtree``
            * ``balltree``
        If you want it to be accurate add:
            * ``slow``

    **kwargs : optional
        Arguments passed to the distance metric

    Returns
    -------
    labels : ndarray, shape (n_samples)
        Cluster labels for each point. Noisy samples are given the label -1.

    min_spanning_tree : ndarray, shape (2*n_samples - 2)
        The minimum spanning tree as edgepairs.

    values_edges : ndarray, shape (n_samples - 1)
        Values of the edges.


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
    printout = ''
    if max_ranking is None:
        max_ranking = 16
        printout += 'max_ranking is set to '+str(max_ranking)+', '

    max_ranking = min(size - 1, max_ranking)

    if limit1 is None:
        limit1 = int(np.sqrt(size))
        printout += 'limit1 is set to '+str(limit1)+', '
    else:
        if type(limit1) is not int:
             raise ValueError('Limit1 must be integer!')
        if limit1 < 0:
            raise ValueError('Limit1 must be non-negative integer!')
    if limit2 is None:
        limit2 = int(size/2 + 1)
        printout += 'limit2 is set to '+str(limit2)+', '
    else:
        if type(limit2) is not int:
             raise ValueError('Limit2 must be integer!')
        if limit2 < 0:
            raise ValueError('Limit2 must be non-negative integer!')

    if algorithm == 'best':
        algorithm = 'kd_tree'

    if X.dtype != np.float64:
        print ('Converting data to numpy float64')
        X = X.astype(np.float64)

    algo_code = 0
    if "precomputed" in algorithm.lower() or "precomputed" in metric.lower() or issparse(X):
        algo_code = 2
        if issparse(X):
            algo_code = 3
        elif len(X.shape)==2 and X.shape[0] != X.shape[1]:
            raise ValueError('Precomputed matrix is not a square.')
        tree = X
    else:
        # The Cython routines used require contiguous arrays
        if not X.flags['C_CONTIGUOUS']:
            X = np.array(X, dtype=np.double, order='C')

        if "kd" in algorithm.lower() and "tree" in algorithm.lower():
            algo_code = 0
            if metric not in KDTree.valid_metrics:
                raise ValueError('Metric: %s\n'
                                 'Cannot be used with KDTree' % metric)
            tree = KDTree(X, metric=metric, leaf_size=leaf_size, **kwargs)
        elif "ball" in algorithm.lower() and "tree" in algorithm.lower():
            algo_code = 1
            tree = BallTree(X, metric=metric, leaf_size=leaf_size, **kwargs)
        else:
            algo_code = 0
            if metric not in KDTree.valid_metrics:
                raise ValueError('Metric: %s\n'
                                 'Cannot be used with KDTree' % metric)
            tree = KDTree(X, metric=metric, leaf_size=leaf_size, **kwargs)
            # raise TypeError('Unknown algorithm type %s specified' % algorithm)

    is_slow_and_deterministic = 0
    if "slow" in algorithm.lower():
        is_slow_and_deterministic = 1

    if printout:
        print ('Druhg is using defaults for: ' + printout)

    ur = UniversalReciprocity(algo_code, tree, max_ranking, metric, leaf_size//3, is_slow_and_deterministic)

    pairs, values = ur.get_tree()

    labels = labeling.label(pairs, values, size, exclude=exclude, limit1=limit1, limit2=limit2, fix_outliers=fix_outliers)

    return (labels,
            pairs, values
            )

class DRUHG(BaseEstimator, ClusterMixin):
    def __init__(self, metric='euclidean',
                 algorithm='best',
                 max_ranking=24,
                 limit1=None,
                 limit2=None,
                 exclude=None,
                 fix_outliers=0,
                 leaf_size=40,
                 verbose=False,
                 **kwargs):
        self.max_ranking = max_ranking
        self.limit1 = limit1
        self.limit2 = limit2
        self.exclude = exclude
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
        self.values_ = None

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
         self.mst_,
         self.values_) = druhg(X, **kwargs)

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

    def relabel(self, exclude=None, limit1=None, limit2=None, fix_outliers=None):
        """Relabeling with the limits on cluster size.

        Parameters
        ----------

        exclude : list of cluster-indexes, for surgical removal of certain clusters,
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
        printout = ''
        size = self._size
        if limit1 is None:
            limit1 = int(np.sqrt(size))
            printout += 'limit1 is set to ' + str(limit1) + ', '
        else:
            if type(limit1) is not int:
                raise ValueError('Limit1 must be integer!')
            if limit1 < 0:
                raise ValueError('Limit1 must be non-negative integer!')

        if limit2 is None:
            limit2 = int(size / 2 + 1)
            printout += 'limit2 is set to ' + str(limit2) + ', '
        else:
            if type(limit2) is not int:
                raise ValueError('Limit2 must be integer!')
            if limit2 < 0:
                raise ValueError('Limit2 must be non-negative integer!')

        if fix_outliers is None:
            fix_outliers = 0
            printout += 'fix_outliers is set to ' + str(fix_outliers)

        if printout:
            print ('Relabeling using defaults for: ' + printout)

        return labeling.label(self.mst_, self.values_, self._size, exclude, limit1, limit2, fix_outliers)

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
