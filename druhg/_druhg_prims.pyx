# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# Minimum spanning tree single linkage implementation for druhg
# Authors: Pavel "DRUHG" Artamonov
# License: 3-clause BSD

# Code to implement a Prims Minimimum Spanning Tree computation
# for even subjective ranking up to max neighbors cap
import numpy as np
cimport numpy as np

from libc.float cimport DBL_MAX
from libc.math cimport fabs, pow

from sklearn.neighbors import KDTree, BallTree

import dist_metrics as dist_metrics
cimport dist_metrics as dist_metrics

from sklearn.externals.joblib import Parallel, delayed

cdef np.double_t INF = np.inf


cdef class MSTPrimsAlgorithm (object):
    """Prims Minimimum Spanning Tree computation using the sklearn
    KDTree/BallTree space tree implementation
    for DRUHG's even subjective ranking distances up to max neighbors

    Parameters
    ----------

    tree : KDTree/BallTree
        The kd-tree to run even subjective ranking over.

    is_kd_tree:
        boolean

    max_neighbors_search : int, optional (default= 16)
        The max_neighbors_search paramater of DRUHG - how many neighbors of the point to rank and even their distances
    min_flatting: int, optional (default=0)
        The min_flatting paramater of DRUHG - how many neighbors of the point to flat rank and even their distances

    metric : string, optional (default='euclidean')
        The metric used to compute distances for the tree

    leaf_size : int, optional (default=20)
        sklearn K-NearestNeighbor uses it

    alpha : float, optional (default=1.0)
        The alpha distance scaling parameter as per Robust Single Linkage.

    **kwargs :
        Keyword args passed to the metric.
    """

    cdef object tree
    cdef object dist_tree
    cdef dist_metrics.DistanceMetric dist
    cdef np.ndarray _data
    cdef np.double_t[:, ::1] _raw_data
    cdef np.double_t[:, :, ::1] node_bounds
    cdef np.double_t alpha
    cdef np.intp_t max_neighbors_search
    cdef np.intp_t min_flatting
    cdef np.intp_t num_points
    cdef np.intp_t num_nodes
    cdef np.intp_t num_features

    cdef public np.double_t[::1] even_rank_distance
    cdef public np.intp_t[::1] candidate_neighbor
    cdef public np.intp_t[::1] candidate_point
    cdef public np.double_t[::1] candidate_distance
    cdef public np.intp_t[::1] idx_array
    cdef np.ndarray edges
    cdef np.intp_t num_edges

    def __init__(self, tree, is_kd_tree, max_neighbors_search=16, min_flatting=1, metric='euclidean', leaf_size=20,
                 alpha=1.0, **kwargs):

        self.dist_tree = tree
        if is_kd_tree:
            self.tree = KDTree(tree.data, metric=metric, leaf_size=leaf_size, **kwargs)
        else:
            self.tree = BallTree(tree.data, metric=metric, leaf_size=leaf_size, **kwargs)
        self._data = np.array(self.tree.data)
        self._raw_data = self.tree.data
        self.node_bounds = self.tree.node_bounds
        self.max_neighbors_search = max_neighbors_search
        self.min_flatting = min_flatting
        self.alpha = alpha

        self.num_points = self.tree.data.shape[0]
        self.num_features = self.tree.data.shape[1]
        self.num_nodes = self.tree.node_data.shape[0]

        self.dist = dist_metrics.DistanceMetric.get_metric(metric, **kwargs)

        self.edges = np.empty((self.num_points - 1, 3))
        self.num_edges = 0

        self._compute_tree()

    cdef _compute_tree(self):
        # DRUHG
        # computes Minimum Spanning Tree for Prims Algorithm
        # for DRUHG's even subjective ranking up to max neighbor
        # for KDTree or BallTree
        # as edges

        cdef np.intp_t c
        cdef np.intp_t best_candidate
        cdef np.intp_t i
        cdef np.intp_t j
        cdef np.intp_t jj
        cdef np.intp_t ki
        cdef np.intp_t kj
        cdef np.intp_t new_idx
        cdef np.intp_t new_candidate
        cdef np.intp_t current_node
        cdef np.double_t dis
        cdef np.double_t max_rank_distance
        cdef np.double_t new_distance
        cdef np.ndarray[np.intp_t, ndim=1] sorted_distances

        cdef np.double_t *raw_data = (<np.double_t *> &self._raw_data[0, 0])
    
        cdef np.ndarray[np.double_t, ndim=2] knn_dist
        cdef np.ndarray[np.intp_t, ndim=2] knn_indices

        cdef np.ndarray[np.double_t, ndim=1] current_distances_arr
        cdef np.ndarray[np.double_t, ndim=1] current_max_rank_distances_arr
        cdef np.ndarray[np.int_t, ndim=1] candidates_arr

        cdef np.ndarray[np.int8_t, ndim=1] in_tree_arr

        cdef np.double_t * current_distances
        cdef np.double_t * current_max_rank_distances
        cdef np.intp_t * candidates

        cdef np.int8_t * in_tree

        current_distances_arr = np.infty * np.ones(self.num_points)
        current_max_rank_distances_arr = np.infty * np.ones(self.num_points)
        candidates_arr = np.ones(self.num_points, dtype=np.int)

        in_tree_arr = np.zeros(self.num_points, dtype=np.int8)

        current_distances = (<np.double_t *> current_distances_arr.data)
        current_max_rank_distances = (<np.double_t *> current_max_rank_distances_arr.data)
        candidates = (<np.intp_t *> candidates_arr.data)

        in_tree = (<np.int8_t *> in_tree_arr.data)

        knn_dist, knn_indices = self.dist_tree.query(
                    self.tree.data,
                    k=self.max_neighbors_search + 1,
                    dualtree=True,
                    breadth_first=True,
                    )

        # druhg
        # Prims algorythm: adding a new node with the smallest edge
    
        # finding smallest even ranking distances for every node
        for i in range(self.num_points):
            current_max_rank_distances[i] = knn_dist[i, self.max_neighbors_search]
            current_distances[i] = -1.


        # Prims search for a new node with a smallest possible edge
        current_node = 0

        sorted_distances = np.argsort(current_distances_arr)

        for r in range(self.num_points-1):

            in_tree[current_node] = 1

            sorted_distances[self.num_points - r - 1] = current_node

            for c in range(self.num_points - r - 1, self.num_points):
                # need to find a node with minimal even ranking distance
                i = sorted_distances[c]

                assert (in_tree[i])
                # if not in_tree[i]:
                #     continue

                new_distance = DBL_MAX
                new_candidate = 0

                max_rank_distance = current_max_rank_distances[i]

                if max_rank_distance >= current_distances[i]:
                    # reevaluate new shortest distance from candidate
                    # checking near neighbors first
                    for ki in range(1, self.max_neighbors_search + 1):
                        if knn_dist[i, ki] >= new_distance:
                            break  # it cannot improve further

                        j = knn_indices[i, ki]
                        if in_tree[j]:
                            continue

                        if knn_dist[i, ki] <= knn_dist[j, ki]:
                            # picking other subjective ranking
                            if ki < self.min_flatting:
                                new_idx = self.min_flatting
                                dis = np.max([knn_dist[j, new_idx], knn_dist[i, new_idx]])
                                if dis < new_distance:
                                    new_distance = dis
                                    new_candidate = j

                            elif knn_dist[j, ki] < new_distance:
                                new_distance = knn_dist[j, ki]
                                new_candidate = j
                        else:
                            # picking it's subjective ranking with higher rank
                            new_idx = -1
                            for kj in range(ki, self.max_neighbors_search + 1):
                                if knn_dist[j, kj] > knn_dist[i, ki]:
                                    break
                                new_idx = kj

                            if new_idx == -1:
                                new_idx = self.max_neighbors_search

                                direct_distance_value = self.dist.dist(&raw_data[self.num_features *
                                                                j],
                                                &raw_data[self.num_features *
                                                                i],
                                                self.num_features)

                                dis = np.max([knn_dist[j, new_idx], max_rank_distance, direct_distance_value])
                                if dis < new_distance:
                                    new_distance = dis
                                    new_candidate = j
                            elif new_idx < self.min_flatting:
                                new_idx = self.min_flatting
                                dis = np.max([knn_dist[j, new_idx], knn_dist[i, new_idx]])
                                if dis < new_distance:
                                    new_distance = dis
                                    new_candidate = j
                            elif knn_dist[i, new_idx] < new_distance:
                                new_distance = knn_dist[i, new_idx]
                                new_candidate = j

                    if new_distance == current_distances[i]:
                        candidates[i] = new_candidate
                        # nothing to improve
                        break
                    if new_distance != DBL_MAX:
                        current_distances[i] = new_distance
                        candidates[i] = new_candidate
                        continue
                    # all near neighbors

                for jj in range(self.num_points - r - 1):
                    j = sorted_distances[jj]
                    if in_tree[j]:
                        continue
                    direct_distance_value = self.dist.dist(&raw_data[self.num_features *
                                                                j],
                                                &raw_data[self.num_features *
                                                                i],
                                                self.num_features)
                    # TODO: multiply by 2? it's the max that can be achieved by even ranking (result of triangle inequality)
                    if self.alpha != 1.0:
                        direct_distance_value /= self.alpha

                    dis = np.max([direct_distance_value, max_rank_distance, current_max_rank_distances[j]])
                    if dis < new_distance:
                        new_distance = dis
                        new_candidate = j

                if new_distance == current_distances[i]:
                    candidates[i] = new_candidate
                    break
                current_distances[i] = new_distance
                candidates[i] = new_candidate

            sorted_distances = np.argsort(current_distances_arr)

            c = self.num_points - r - 1
            current_node = sorted_distances[c]

            if in_tree[candidates[current_node]]:
                # print ('equal distances', i, current_distances[i], current_distances[sorted_distances[c+1]])
                jj = c + 1
                while in_tree[candidates[sorted_distances[jj]]]:
                    jj += 1
                swap = sorted_distances[jj]
                sorted_distances[jj] = current_node
                current_node = swap

            assert (in_tree[current_node])
            assert (not in_tree[candidates[current_node]])

            self.edges[r, 0] = current_node
            self.edges[r, 1] = candidates[current_node]
            self.edges[r, 2] = current_distances[current_node]
            current_node = candidates[current_node]


            # print self.edges[r, 0], self.edges[r, 1], self.edges[r, 2]

    def spanning_tree(self):
        """Returns tree"""

        return self.edges
