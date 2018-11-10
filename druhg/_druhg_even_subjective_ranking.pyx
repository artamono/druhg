# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# Minimum spanning tree single linkage implementation for druhg
# Authors: Pavel "DRUHG" Artamonov
# License: 3-clause BSD

# Code to implement a Prims Minimimum Spanning Tree computation
# with max neighbors cap
import numpy as np
cimport numpy as np

cdef np.double_t INF = np.inf
from libc.math cimport fabs, pow

from sklearn.neighbors import KDTree, BallTree

import dist_metrics as dist_metrics
cimport dist_metrics as dist_metrics

from sklearn.externals.joblib import Parallel, delayed

cdef class EvenSubjectiveRanking (object):
    """Evaluates edges in minimal spanning tree depending on the chosen mode

    Parameters
    ----------

    tree : KDTree/BallTree
        The kd-tree to run even subjective ranking over.

    is_kd_tree:
        boolean

    max_neighbors_search : int, optional (default= 16)
        The max_neighbors_search paramater of DRUHG - how many neighbors of the point to rank and even their distances

    metric : string, optional (default='euclidean')
        The metric used to compute distances for the tree

    leaf_size : int, optional (default=20)
        sklearn K-NearestNeighbor uses it

    **kwargs :
        Keyword args passed to the metric.
    """

    cdef object tree
    cdef object dist_tree
    cdef dist_metrics.DistanceMetric dist
    cdef np.ndarray _data
    cdef np.double_t[:, ::1] _raw_data
    cdef np.double_t[:, :, ::1] node_bounds

    cdef np.intp_t num_points
    cdef np.intp_t num_nodes
    cdef np.intp_t num_features

    cdef np.intp_t num_edges

    cdef np.intp_t max_neighbors_search
    cdef np.intp_t min_flatting

    cdef np.ndarray result_edge

    def __init__(self, tree, is_kd_tree, max_neighbors_search=16, metric='euclidean', leaf_size=20, **kwargs):

        self.dist_tree = tree
        if is_kd_tree:
            self.tree = KDTree(tree.data, metric=metric, leaf_size=leaf_size, **kwargs)
        else:
            self.tree = BallTree(tree.data, metric=metric, leaf_size=leaf_size, **kwargs)
        self._data = np.array(self.tree.data)
        self._raw_data = self.tree.data
        self.node_bounds = self.tree.node_bounds
        self.max_neighbors_search = max_neighbors_search

        self.dist = dist_metrics.DistanceMetric.get_metric(metric, **kwargs)

        self.num_points = self.tree.data.shape[0]
        self.num_features = self.tree.data.shape[1]
        self.num_nodes = self.tree.node_data.shape[0]
        self.num_edges = self.num_points - 1

        self.result_edge = np.empty((self.num_points))

        self._compute_tree_edges()

    cdef _compute_tree_edges(self):
        # DRUHG
        # computes Minimum Spanning Tree for Prims Algorithm
        # for KDTree or BallTree
        # the evaluated weight-distances are don't matter, only edges are

        cdef np.intp_t r
        cdef np.intp_t c
        cdef np.intp_t i
        cdef np.intp_t j
        cdef np.intp_t jj
        cdef np.intp_t ki
        cdef np.intp_t kj
        cdef np.intp_t new_node
        cdef np.intp_t old_node
        cdef np.double_t dis
        cdef np.double_t max_rank_distance
        cdef np.double_t new_distance
        cdef np.ndarray[np.intp_t, ndim=1] sorted_distances
        cdef np.ndarray[np.double_t, ndim=1] current_distances_arr

        cdef np.double_t *raw_data = (<np.double_t *> &self._raw_data[0, 0])
        cdef np.double_t * current_distances

        cdef np.ndarray[np.double_t, ndim=2] knn_dist
        cdef np.ndarray[np.intp_t, ndim=2] knn_indices

        current_distances_arr = np.infty * np.ones(self.num_points)

        current_distance = (<np.double_t *> current_distances_arr.data)

        knn_dist, knn_indices = self.dist_tree.query(
                    self.tree.data,
                    k=self.max_neighbors_search + 1,
                    dualtree=True,
                    breadth_first=True,
                    )

        # Preparation

        # adding first node
        old_node = self.num_edges

        self.result_edge[old_node] = -1
        for i in range(self.num_points):
            current_distance[i] = -1.

        # Prims search for a new node with a smallest possible edge
        for r in range(self.num_edges):

            current_distance[old_node] = 0.
            sorted_distances = np.argsort(current_distances_arr)

            new_node = -1
            # sorted_distances[self.num_edges - r] = old_node
            if sorted_distances[self.num_edges - r] != old_node:
                print (sorted_distances[self.num_edges - r], old_node)

            assert(sorted_distances[self.num_edges - r] == old_node)

            for c in range(self.num_edges - r, self.num_points):
                # iterating other already used nodes
                # need to find another node with minimal distance to the existing tree
                new_node = -1

                i = sorted_distances[c]

                assert (i>=0)
                assert (current_distance[i]>=0.)

                max_rank_distance = knn_dist[i, self.max_neighbors_search]

                new_distance = INF

                # reevaluate new shortest distance from candidate
                if max_rank_distance >= current_distance[i]:
                    # checking near neighbors first
                    for ki in range(1, self.max_neighbors_search + 1):
                        if knn_dist[i, ki] >= new_distance:
                            break  # it cannot improve further

                        j = knn_indices[i, ki]
                        # print(r,c,'knn',j, current_distance[j])

                        if current_distance[j]>=0.:
                            continue
                        old_node = i
                        new_node = j
                        new_distance = knn_dist[j, ki]
                        break

                # all near neighbors are used
                if new_distance > max_rank_distance:
                    for jj in range(self.num_edges - r):
                        j = sorted_distances[jj]
                        assert (j>=0)
                        if current_distance[j] != -1.:
                            print (r,c,new_distance, i,j, self.num_edges - r - 1, jj, self.num_edges - r, current_distance[j] == -1., current_distance[j])
                        assert (current_distance[j] == -1.)

                        dis = self.dist.dist(&raw_data[self.num_features *
                                                                    j],
                                                    &raw_data[self.num_features *
                                                                    i],
                                                    self.num_features)
                        if dis < new_distance:
                            new_distance = dis
                            new_node = j

                old_node = i
                if new_distance == current_distance[i]:
                    # nothing to improve
                    break

                current_distance[old_node] = new_distance

            # new node is found

            # print('result', new_node, old_node, current_distance[new_node], current_distance[old_node])
            assert (new_node!=-1)
            assert (current_distance[new_node]==-1.)
            assert (current_distance[old_node]>=0.)

            self.result_edge[new_node] = old_node
            old_node = new_node

    def spanning_tree_edges(self):
        """Returns tree. One of the nodes is not connected and has -1"""

        return self.result_edge

    def distances(self, mode='hybrid', min_ranking=0, max_ranking=1, step_ranking=0):
        """"Returns distances-weights of the evaluated edges.
            DRUHG algorithm
            Depends on the mode"""

        cdef np.intp_t r
        cdef np.intp_t c
        cdef np.intp_t i
        cdef np.intp_t j
        cdef np.intp_t jj
        cdef np.intp_t ki
        cdef np.intp_t kj
        cdef np.intp_t new_node
        cdef np.intp_t old_node
        cdef np.double_t dis
        cdef np.double_t max_rank_distance
        cdef np.double_t new_distance
        cdef np.ndarray[np.double_t, ndim=1] result_distances_arr

        cdef np.double_t *raw_data = (<np.double_t *> &self._raw_data[0, 0])
        cdef np.double_t * result_distance

        cdef np.ndarray[np.double_t, ndim=2] knn_dist
        cdef np.ndarray[np.intp_t, ndim=2] knn_indices

        result_distances_arr = np.infty * np.ones(self.num_points)
        result_distance = (<np.double_t *> result_distances_arr.data)

        knn_dist, knn_indices = self.dist_tree.query(
                    self.tree.data,
                    k=max_ranking + 1,
                    dualtree=True,
                    breadth_first=True,
                    )

        for old_node in range(self.num_edges):

            new_node = self.result_edge[old_node]

            assert (new_node >= 0)

            dis = self.dist.dist(&raw_data[self.num_features *
                                                        old_node],
                                        &raw_data[self.num_features *
                                                        new_node],
                                        self.num_features)

            # need to find each other ranks in each others KNN queries
            rank_i = rank_j = max_ranking

            for ki in range(1, max_ranking + 1):
                if knn_dist[old_node, ki] <= dis:
                    rank_i = ki
                if knn_dist[new_node, ki] <= dis:
                    rank_j = ki

            if rank_i < min_ranking:
                rank_i = min_ranking
            if rank_j < min_ranking:
                rank_j = min_ranking

            rank_i += step_ranking
            rank_j += step_ranking

            if rank_i > max_ranking:
                rank_i = max_ranking
            if rank_j > max_ranking:
                rank_j = max_ranking

            rank_min = min(rank_i, rank_j)
            rank_max = max(rank_i, rank_j)

            if mode == 'distance':
                dis = dis
            elif mode == 'distance_max':
                dis = max(knn_dist[old_node,rank_max],knn_dist[new_node,rank_max],dis)
            elif mode == 'distance_diff':
                dis = max(knn_dist[old_node,rank_max],knn_dist[new_node,rank_max],dis) - dis
            elif mode == 'ranks':
                dis = rank_i + rank_j
            elif mode == 'ranks_max':
                dis = rank_max
            elif mode == 'ranks_diff':
                dis = 1 + rank_max - rank_min
            elif mode == 'hybrid':
                dis = dis*(1 + rank_max - rank_min)

            result_distance[old_node] = dis

        return result_distances_arr
