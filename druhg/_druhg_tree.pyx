# cython: boundscheck=False
# cython: nonecheck=False
# cython: initializedcheck=False
# DRUHG
#
# Authors: Pavel "DRUHG" Artamonov
# License: 3-clause BSD

import numpy as np
cimport numpy as np

import bisect

import sys

import gc

cdef np.double_t INF = np.inf
cdef np.double_t EPS = np.finfo(type(INF)).eps

cdef np.intp_t MAX_INT = np.iinfo(int).max

from sklearn.neighbors import KDTree, BallTree

import dist_metrics as dist_metrics
cimport dist_metrics as dist_metrics

cdef class Tree(object):
    """Builds and evaluates edges in spanning tree depending on the chosen mode

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

    cdef np.intp_t num_points, num_nodes, num_features

    cdef np.intp_t max_neighbors_search

    # union find structure
    cdef np.ndarray parent_arr
    cdef np.intp_t next_label
    cdef np.intp_t *parent

    # result tree
    cdef np.intp_t num_edges

    cdef np.ndarray edge_node_pairs_arr
    cdef np.intp_t *edge_node_pairs

    cdef np.ndarray edge_order_arr
    cdef np.double_t *edge_order

    cdef np.ndarray edge_data_arr
    cdef np.ndarray edge_data_arr2

    cdef np.double_t *edge_rr_dis
    cdef np.intp_t *edge_max_rank
    # not used
    cdef np.intp_t *edge_min_rank
    cdef np.double_t *edge_org_dis

    def __init__(self, tree, is_kd_tree, max_neighbors_search=16, metric='euclidean', leaf_size=20, **kwargs):

        self.dist_tree = tree
        if is_kd_tree:
            self.tree = KDTree(tree.data, metric=metric, leaf_size=leaf_size, **kwargs)
        else:
            self.tree = BallTree(tree.data, metric=metric, leaf_size=leaf_size, **kwargs)
        self._data = np.array(self.tree.data)
        self.max_neighbors_search = max_neighbors_search

        self.dist = dist_metrics.DistanceMetric.get_metric(metric, **kwargs)

        self.num_points = self.tree.data.shape[0]
        self.num_features = self.tree.data.shape[1]
        self.num_nodes = self.tree.node_data.shape[0]

        # union find
        self.next_label = self.num_points
        self.parent_arr = np.zeros(2 * self.num_points, dtype=np.intp)
        self.parent = (<np.intp_t *> self.parent_arr.data)

        # self.result_edges_arr = np.empty((self.num_points))
        # self.result_edges = (<np.intp_t *> result_edges_arr.data)

        # result tree
        self.num_edges = 0

        self.edge_node_pairs_arr = np.zeros(2 * self.num_points - 2, dtype=np.intp)
        self.edge_node_pairs = (<np.intp_t *> self.edge_node_pairs_arr.data)

        self.edge_order_arr = np.zeros(self.num_points - 1)
        self.edge_order = (<np.double_t *> self.edge_order_arr.data)

        self.edge_data_arr = np.zeros(self.num_points - 1)
        self.edge_data_arr2 = np.zeros(self.num_points - 1, dtype=np.intp)

        self.edge_rr_dis = (<np.double_t *> self.edge_data_arr.data)
        self.edge_max_rank = (<np.intp_t *> self.edge_data_arr2.data)

        # self.edge_org_dis = (<np.double_t *> self.edge_data_arr.data)
        # self.edge_min_rank = (<np.intp_t *> self.edge_data_arr2.data)

        self._compute_tree_edges_knn_limited()

    def get_edges_flat(self):
        """Returns edges of the tree. Node pairs and count."""

        return (self.edge_node_pairs_arr, self.num_edges)

    def get_order(self):
        """Returns reciprocal order for edges."""

        return self.edge_order_arr

    def get_data(self):
        """Returns data for edges.
           Reciprocal rank distances and ranks."""
        return (self.edge_data_arr, self.edge_data_arr2)

    # union find
    cdef np.intp_t fast_find(self, np.intp_t n):
        cdef np.intp_t p, t
        assert (self.parent[n] != 0)
        p = n
        while self.parent[p] != 0:
            p = self.parent[p]
        # label up to the root
        while self.parent[n] != p:
            t = self.parent[n]
            self.parent[n] = p
            n = t
        return p

    cdef np.intp_t has_cycle(self, np.intp_t node1, np.intp_t node2):
        if not self.parent[node1] or not self.parent[node2]:
            return 0
        return self.fast_find(node1) == self.fast_find(node2)

    cdef unite_nodes(self, np.intp_t node1, np.intp_t node2):
        cdef np.intp_t par1, par2
        par1, par2 = self.parent[node1], self.parent[node2]
        if par1:
            par1 = self.fast_find(node1)
        else:
            par1 = node1

        if par2:
            par2 = self.fast_find(node2)
        else:
            par2 = node2

        self.parent[par1], self.parent[par2] = self.next_label, self.next_label
        self.next_label += 1

    cdef unite_node_parents(self, np.intp_t par1, np.intp_t par2):
        self.parent[par1], self.parent[par2] = self.next_label, self.next_label
        self.next_label += 1

    # result: data and edges
    cdef add_edge_data(self, np.intp_t index, np.double_t rr_dis, np.double_t org_dis, np.intp_t max_rank,
                       np.intp_t min_rank, np.double_t order):
        self.edge_rr_dis[index] = rr_dis
        # self.edge_org_dis[index] = org_dis
        self.edge_max_rank[index] = max_rank + 1
        # self.edge_min_rank[index] = min_rank
        self.edge_order[index] = order

    cdef add_node_pair(self, np.intp_t index, np.intp_t node1, np.intp_t node2):
        self.edge_node_pairs[2 * index] = node1
        self.edge_node_pairs[2 * index + 1] = node2

    # algorythm
    cdef _compute_tree_edges_knn_limited(self):
        # DRUHG
        # computes k-Limited Reciprocal Spanning Tree/Forest
        # for KDTree or BallTree
        # Tree is limited by max_neighbors and might be not complete

        cdef np.intp_t i, j, ki, s, parent, relatives
        cdef np.intp_t lim_rank, opp_rank, min_rank, max_rank
        cdef np.intp_t opt_r, opt_R, opt_edge
        cdef np.double_t opt_dis, opt_org
        cdef np.double_t global_opt, opt, dis, lim_dis

        cdef set curobjs
        cdef tuple global_data

        cdef np.ndarray[np.intp_t, ndim=1] sorted_distances_arr
        cdef np.intp_t *sorted_distances

        cdef np.ndarray[np.double_t, ndim=1] current_distances_arr
        cdef np.double_t *current_distance
        current_distances_arr = np.zeros(self.num_points)
        current_distance = (<np.double_t *> current_distances_arr.data)

        cdef np.ndarray[np.double_t, ndim=2] knn_dist
        cdef np.ndarray[np.intp_t, ndim=2] knn_indices
        knn_dist, knn_indices = self.dist_tree.query(
            self.tree.data,
            k=self.max_neighbors_search + 1,
            dualtree=True,
            breadth_first=True,
        )

        # todo: improve description
        # node N has k nearest neighbors and those neighbors might have this node N as their k nearest neighbor
        # algorithm will build the spanning tree trying to minimize max( neighbor rank, it's reciprocal rank) and then their min()
        # and then their original distance (skipping the latter)

        opt_dis, opt_org = 0., 0.
        opt_r, opt_R, opt_edge = -1, -1, -1
        curobjs = set()

        # finding the optimal reciprocal edge for every node
        for i in range(0, self.num_points):

            opt = -1.
            curobjs.clear()

            lim_dist, lim_rank = 0., 1
            for ki in range(0, self.max_neighbors_search + 1):
                j = knn_indices[i, ki]
                curobjs.add(j)
                if j == i:
                    continue

                dis = knn_dist[i, ki]
                if lim_dist < dis:
                    lim_dist, lim_rank = dis, ki

                # finding reciprocal rank for equal distance
                opp_rank = bisect.bisect_left(knn_dist[j], dis)

                relatives = len(curobjs.intersection(knn_indices[j][:opp_rank + 1])) - 1

                if lim_rank >= opp_rank:
                    rank_dis = knn_dist[j, lim_rank]
                    rank_max, rank_min = lim_rank, opp_rank
                elif opp_rank <= self.max_neighbors_search:
                    rank_dis = knn_dist[i, opp_rank]
                    rank_max, rank_min = opp_rank, lim_rank
                else:  # adding penalty
                    rank_dis = knn_dist[i, opp_rank - 1]
                    rank_max, rank_min = opp_rank + rank_min, lim_rank

                # this can be changed
                order = 1. / rank_dis * (rank_min + 1) / (rank_max + relatives)

                # print(i, lim_rank, j, round(order,4), 'data', dis, rank_dis, rank_min, rank_max, relatives,
                #       'ratio', 1./rank_dis, 1./rank_max, (rank_min + 1)/(rank_max + relatives), (dis/rank_dis))

                if order > opt:
                    opt, opt_edge = order, j
                    opt_org, opt_dis = dis, rank_dis
                    opt_r, opt_R = rank_min, rank_max

                if 1. / dis <= opt:
                    break

            # todo: we need to increase value of low-rank relations
            # todo: need to deal with batches of equal opts

            current_distance[i] = opt
            if not self.has_cycle(i, opt_edge):
                self.unite_nodes(i, opt_edge)
                self.add_edge_data(self.num_edges, opt_dis, opt_org, opt_R, opt_r, opt)
                self.add_node_pair(self.num_edges, i, opt_edge)
                self.num_edges += 1
                # print('-=========', i, opt_edge, 'opt', opt_dis, opt_org, opt_R, opt_r, opt)
            # else:
            #     print ('cycle', i,opt_edge)

        ###############################
        # about 70% of links are found
        # building a tree one best link at a time
        ###############################

        # print ('edges', self.num_edges, self.num_points - self.num_edges, 1.*self.num_edges/self.num_points)

        while self.num_edges <= self.num_points - 2:
            sorted_distances_arr = np.argsort(current_distances_arr)
            sorted_distances = (<np.intp_t *> sorted_distances_arr.data)

            global_opt = -1.
            s = self.num_points
            while s:
                s -= 1
                i = sorted_distances[s]

                if current_distance[i] <= global_opt:
                    break
                parent = self.fast_find(i)
                curobjs.clear()

                opt = -1.
                lim_dist, lim_rank = 0., 1
                for ki in range(0, self.max_neighbors_search + 1):
                    j = knn_indices[i, ki]
                    curobjs.add(j)
                    if parent == self.fast_find(j):
                        continue

                    dis = knn_dist[i, ki]
                    if lim_dist < dis:
                        lim_dist, lim_rank = dis, ki

                    # finding reciprocal rank for equal distance
                    opp_rank = bisect.bisect_left(knn_dist[j], dis)

                    relatives = len(curobjs.intersection(knn_indices[j][:opp_rank + 1])) - 1

                    if lim_rank >= opp_rank:
                        rank_dis = knn_dist[j, lim_rank]
                        rank_max, rank_min = lim_rank, opp_rank
                    elif opp_rank <= self.max_neighbors_search:
                        rank_dis = knn_dist[i, opp_rank]
                        rank_max, rank_min = opp_rank, lim_rank
                    else:  # adding penalty
                        rank_dis = knn_dist[i, opp_rank - 1]
                        rank_max, rank_min = opp_rank + rank_min, lim_rank

                    order = 1. / rank_dis * (rank_min + 1) / (rank_max + relatives)

                    # print(i, j, 'order2', i, order, dis/rank_dis,1.*rank_min/rank_max,1.*(1+len(curobjs.difference(allobjs)))/(2*rank_max), 'ranks',rank_min,rank_max,len(curobjs.difference(allobjs)), len(curobjs))

                    if order > opt:
                        opt, opt_edge = order, j
                        opt_org, opt_dis = dis, rank_dis
                        opt_r, opt_R = rank_min, rank_max

                    if 1. / dis <= opt:
                        break

                current_distance[i] = opt

                if opt > global_opt:
                    global_opt = opt
                    global_data = (i, opt_edge, opt_dis, opt_org, opt_R, opt_r)

            if global_opt > 0.:
                i, opt_edge, opt_dis, opt_org, opt_R, opt_r = global_data
                self.unite_node_parents(self.fast_find(i), self.fast_find(opt_edge))

                self.add_edge_data(self.num_edges, opt_dis, opt_org, opt_R, opt_r, global_opt)
                self.add_node_pair(self.num_edges, i, opt_edge)

                self.num_edges += 1
            else:
                break

        if self.num_edges + 1 != self.num_points:
            print ('Forest is build instead of a tree.' + str(
                1 * self.num_edges / (self.num_points - 1)) + '%. Missing ' + str(
                self.num_points - self.num_edges - 1) + ' edges. You can increase productivity parameter ' + str(
                self.max_neighbors_search) + ' to increase precision.')

        return
