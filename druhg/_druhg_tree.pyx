# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# builds minimum spanning tree for druhg algorithm
# Author: Pavel "DRUHG" Artamonov
# License: 3-clause BSD


import numpy as np
cimport numpy as np

import pandas as pd

from libc.stdlib cimport malloc, free

from scipy import stats

cdef np.double_t INF = np.inf

from libc.math cimport fabs, pow

from sklearn.neighbors import KDTree, BallTree
from sklearn import preprocessing

import bisect

from sklearn.externals.joblib import Parallel, delayed

cdef np.double_t merge_means(np.intp_t na, np.double_t meana,
                             np.intp_t nb, np.double_t meanb
                            ):
# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
# Chan et al.[10] Welford's online algorithm
    cdef np.double_t delta

    # nx = na + nb
    delta = meanb - meana
    meana = meana + delta*nb/(na + nb)
    # use this for big n's
    # mu = (mu*n + mu_2*n_2) / nx
    # m2a = m2a + m2b + delta**2*na*nb/nx
    return meana

cdef class PairwiseDistanceTreeSparse(object):
    cdef object data_arr
    cdef int data_size

    def __init__(self, N, d):
        self.data_size = N
        self.data_arr = d

    cpdef tuple query(self, d, k, dualtree = 0, breadth_first = 0):
        cdef np.ndarray[np.double_t, ndim=2] knn_dist
        cdef np.ndarray[np.intp_t, ndim=2] knn_indices

        knn_dist = INF*np.ones((self.data_size, k))
        knn_indices = np.zeros((self.data_size, k), dtype=np.intp)

        i = self.data_size
        while i:
            i -= 1
            row = self.data_arr.getrow(i)
            idx, data = row.indices, row.data
            sorted = np.argsort(data)
            j = min(k,len(idx))
            if idx[sorted[0]] != i:
                while j:
                    j -= 1
                    knn_dist[i][j+1] = data[sorted[j]]
                    knn_indices[i][j+1] = idx[sorted[j]]
                # have to add itself
                knn_dist[i][0], knn_indices[i][0] = 0., i
            else:
                while j:
                    j -= 1
                    knn_dist[i][j] = data[sorted[j]]
                    knn_indices[i][j] = idx[sorted[j]]

        return knn_dist, knn_indices

cdef class PairwiseDistanceTreeGeneric(object):
    cdef object data_arr
    cdef int data_size

    def __init__(self, N, d):
        self.data_size = N
        self.data_arr = d

    cpdef tuple query(self, d, k, dualtree = 0, breadth_first = 0):
        cdef np.ndarray[np.double_t, ndim=2] knn_dist
        cdef np.ndarray[np.intp_t, ndim=2] knn_indices

        knn_dist = np.zeros((self.data_size, k))
        knn_indices = np.zeros((self.data_size, k), dtype=np.intp)

        i = self.data_size
        while i:
            i -= 1
            row = self.data_arr[i]
            sorted = np.argsort(row)
            j = k
            while j:
                j -= 1
                knn_dist[i][j] = row[sorted[j]]
                knn_indices[i][j] = sorted[j]

        return knn_dist, knn_indices

cdef class UnionFindMST (object):

    cdef np.ndarray parent_arr
    cdef np.ndarray size_arr
    cdef np.ndarray energy_arr

    cdef np.intp_t *parent
    cdef np.intp_t *size
    cdef np.double_t *energy

    cdef np.intp_t full_size
    cdef np.intp_t next_label

    def __init__(self, N):
        self.full_size = N
        self.next_label = N + 1

        self.parent_arr = np.zeros(2 * N, dtype=np.intp)
        self.size_arr = np.ones(N, dtype=np.intp)
        self.energy_arr = 0.*np.zeros(N)

        self.parent = (<np.intp_t *> self.parent_arr.data)
        self.size = (<np.intp_t *> self.size_arr.data)
        self.energy = (<np.double_t *> self.energy_arr.data)

    cdef np.intp_t fast_find(self, np.intp_t n):
        cdef np.intp_t p, temp

        p = self.parent[n]
        if p == 0:
            return n
        while self.parent[p] != 0:
            p = self.parent[p]

        # label up to the root
        while p != n:
            temp = self.parent[n]
            self.parent[n] = p
            n = temp

        return p

    cdef np.intp_t is_cluster(self, np.intp_t n):
        return self.parent[n]

    cdef void union(self, np.intp_t aa, np.intp_t bb):
        self.parent[aa] = self.parent[bb] = self.next_label
        self.next_label += 1
        return

    cdef list reciprocal_emergence_of_energy_of_commonalities(self, np.intp_t a, np.intp_t b):
        cdef np.intp_t aa, bb, na, nb, i
        cdef np.double_t dip, old_energy, new_energy, excess_of_energy
        cdef list ret

        ret = []
        i = self.next_label - self.full_size
        aa, bb = self.fast_find(a), self.fast_find(b)

        a = (self.parent[a] != 0)*(aa - self.full_size)
        b = (self.parent[b] != 0)*(bb - self.full_size)
        self.union(aa, bb) # changes parents

        na, nb = self.size[a], self.size[b]
        self.size[i] = na + nb
        dip = 1./np.sqrt(min(na, nb))
# ----------------------
        new_energy = 0.
        old_energy = self.energy[a]
        excess_of_energy = 1. - dip - old_energy
        if excess_of_energy > old_energy:
            new_energy = excess_of_energy
            ret.append(a + self.full_size)
        else:
            new_energy = old_energy
# ----------------------
        old_energy = self.energy[b]
        excess_of_energy = 1. - dip - old_energy
        if excess_of_energy > old_energy:
            new_energy = merge_means(na, new_energy, nb, excess_of_energy)
            ret.append(b + self.full_size)
        else:
            new_energy = merge_means(na, new_energy, nb, old_energy)
# ----------------------
        self.energy[i] = new_energy
        return ret

    cdef np.double_t get_last_energy(self):
        cdef np.intp_t i, p, na, nb
        cdef np.double_t old_energy

        i = self.next_label - 1
        old_energy = self.energy[i - self.full_size]
        na = self.size[i - self.full_size]

        # if 1. > 2. * old_energy + dip where dip is 0. or 1/sqrt(size)

        if na == self.full_size:
            return old_energy

        # it is a forest
        parents = set()
        parents.update([i])
        i = self.full_size
        while i != 0:
            i -= 1
            p = self.fast_find(i)
            if p not in parents:
                parents.update([p])
                nb = self.size[p - self.full_size]
                old_energy = merge_means(na, old_energy, nb, self.energy[p - self.full_size])
                na += nb
                if na >= self.full_size:
                    break

        return old_energy

    cdef np.intp_t universal_completeness(self):
        # does this make sense or dip is 0?
        cdef np.intp_t i
        cdef np.double_t dip, old_energy, excess_of_energy

        i = self.next_label - self.full_size - 1
        dip = 1./np.sqrt(self.full_size)
        old_energy = self.energy[i]
        excess_of_energy = 1. - dip - old_energy
        # print (excess_of_energy > old_energy, 'universal_completeness', excess_of_energy, dip, old_energy, 'size', self.size[i])
        if excess_of_energy > old_energy:
            return 1
        else:
            return 0

    cdef np.intp_t get_new_size(self, np.intp_t a, np.intp_t b):

        a = (self.parent[a] != 0)*(self.fast_find(a) - self.full_size)
        b = (self.parent[b] != 0)*(self.fast_find(b) - self.full_size)

        return self.size[a] + self.size[b]


cdef class UniversalReciprocity (object):
    """Constructs DRUHG spanning tree and marks parents of clusters

    Parameters
    ----------

    algorithm : int
        0/1 - for KDTree/BallTree object
        2/3 - for a full/scipy.sparse precomputed pairwise squared distance matrix

    data: object
        Pass KDTree/BallTree objects or pairwise matrix.

    max_neighbors_search : int, optional (default= 16)
        The max_neighbors_search parameter of DRUHG.
        Effects performance vs precision.
        Default is more than enough.

    metric : string, optional (default='euclidean')
        The metric used to compute distances for the tree.
        Used only with KDTree/BallTree option.

    leaf_size : int, optional (default=20)
        sklearn K-NearestNeighbor uses it.
        Used only with KDTree/BallTree option.

    **kwargs :
        Keyword args passed to the metric.
        Used only with KDTree/BallTree option.
    """

    cdef object tree
    cdef object dist_tree

    cdef np.intp_t num_points
    cdef np.intp_t num_features

    cdef np.intp_t max_neighbors_search

    cdef UnionFindMST U
    cdef np.double_t result_clustered_energy
    cdef np.intp_t result_edges
    cdef np.intp_t result_clusters
    cdef np.ndarray result_value_arr
    cdef np.ndarray result_pairs_arr

    cdef np.double_t *result_value
    cdef np.intp_t *result_pairs

    def __init__(self, algorithm, tree, max_neighbors_search=16, metric='euclidean', leaf_size=20, **kwargs):

        if algorithm == 0:
            self.dist_tree = tree
            self.tree = KDTree(tree.data, metric=metric, leaf_size=leaf_size, **kwargs)
            self.num_points = self.tree.data.shape[0]
        elif algorithm == 1:
            self.dist_tree = tree
            self.tree = BallTree(tree.data, metric=metric, leaf_size=leaf_size, **kwargs)
            self.num_points = self.tree.data.shape[0]
        elif algorithm == 2:
            self.dist_tree = PairwiseDistanceTreeGeneric(tree.shape[0], tree)
            self.tree = tree
            self.num_points = self.tree.shape[0]
        elif algorithm == 3:
            self.dist_tree = PairwiseDistanceTreeSparse(tree.shape[0], tree)
            self.tree = tree
            self.num_points = self.tree.shape[0]
        else:
            raise ValueError('algorithm value '+str(algorithm)+' is not valid')

        self.max_neighbors_search = max_neighbors_search

        # self.num_features = self.tree.data.shape[1]

        self.U = UnionFindMST(self.num_points)

        self.result_clustered_energy = 0.
        self.result_edges = 0
        self.result_clusters = 0

        self.result_pairs_arr = np.empty((self.num_points*2 - 2))
        self.result_value_arr = np.empty(int(self.num_points/2) + 1)

        self.result_pairs = (<np.intp_t *> self.result_pairs_arr.data)
        self.result_value = (<np.double_t *> self.result_value_arr.data)

        self._compute_tree_edges()

    cpdef tuple get_tree(self):
        return (self.result_edges, self.result_pairs_arr.astype(int))

    cpdef tuple get_clusters_parents(self):
        return (self.result_clusters, self.result_value_arr[:self.result_clusters])

    cdef void result_add_edge(self, np.intp_t a, np.intp_t b):
        cdef np.intp_t i

        i = self.result_edges
        self.result_pairs_arr[2*i] = a
        self.result_pairs_arr[2*i + 1] = b
        self.result_edges += 1

    cdef void result_add_cluster(self, np.intp_t p):
        self.result_value[self.result_clusters] = p
        self.result_clusters += 1

    cdef np.intp_t _pure_reciprocity(self, i, knn_indices, knn_dist):
        parent = self.U.fast_find(i)
        indices, distances = knn_indices[i], knn_dist[i]
        for ranki in range(0, 2):
            j = indices[ranki]
            if parent == self.U.fast_find(j):
                continue

            dis = distances[ranki]
            if dis == 0.: # degenerate case
                return j

            rank_left = bisect.bisect(distances, dis)
            if rank_left > 2:
                return -1

            dis_opp = knn_dist[j]
            rank_right = bisect.bisect(dis_opp, dis)
            if rank_right > 2:
                return -1
            return j
        return -1

    cdef tuple _evaluate_reciprocity(self, i, knn_indices, knn_dist, start_rank = 0):
        parent = self.U.fast_find(i)
        indices, distances = knn_indices[i], knn_dist[i]

        opt, opt_min = INF, INF
        opt_edge = 0
        opt_rank = self.max_neighbors_search
        val1, val2, val3 = 1., 1, 1
        opt_dump = ()

        for ranki in range(start_rank, self.max_neighbors_search + 1):
            j = indices[ranki]
            if parent == self.U.fast_find(j):
                continue

            dis = distances[ranki]

            if dis**4*ranki > opt:
                break

            dis_opp = knn_dist[j]
            rank_right = bisect.bisect(dis_opp, dis) # reminder that bisect.bisect(dis_opp, dis) >= bisect.bisect_left(dis_opp, dis)
            if ranki > rank_right:
                continue
            rank_left = bisect.bisect(distances, dis)
            if rank_left > rank_right:
                continue

            scope_size = rank_right - 1
            rank_dis = distances[scope_size]

            ind_opp = knn_indices[j]
            parent_opp = self.U.fast_find(j)

            members = 0
            opp_is_reachable = 0
            for k, s in enumerate(ind_opp[:rank_right]):
                p = self.U.fast_find(s)
                members += parent_opp==p
                opp_is_reachable += parent==p

            penalty = 0 # penalizing in case of reaching the limit of max_neighbors_search
            if opp_is_reachable == 0: # rank_right >= self.max_neighbors_search:
                penalty = rank_left

            val1 = rank_dis # без этого не отличить углов от ребер в квадрате
            val2 = scope_size + penalty # без этого не различить ядро квадрата от ребер
            val3 = members # без этого не обеспечить равномерного прирастание

            order = val1**4
            order_min = order*val2
            order = order_min*val2/val3

            if order_min < opt_min:
                opt_min = order_min

            if order < opt:
                opt, opt_edge = order, j
                opt_rank = scope_size
            # print (i, j, 'vals', val1, val2, val3, 'opts', opt, opt_edge, opt_rank, opt_min)

        return (opt, opt_edge, opt_rank, opt_min)

    cdef _compute_tree_edges(self, start_rank = 0):
        # DRUHG
        # computes DRUHG Spanning Tree

        cdef np.intp_t i, j, ranki, rankj, \
            rank_min, rank_max, lim_rank, \
            relatives, \
            opti, optj, \
            s
        cdef np.double_t dis, rank_dis, lim_dis, \
            opt, global_opt, new_opt, \
            opt1, opt2, val1, val2
        cdef set objs
        cdef np.ndarray[np.double_t, ndim=1] starting_value_arr
        cdef np.ndarray[np.intp_t, ndim=1] sorted_value_arr

        # cdef np.double_t *raw_data = (<np.double_t *> &self._raw_data[0, 0])
        cdef np.double_t * starting_value
        cdef np.intp_t * sorteds

        starting_value_arr = -1.*np.ones(self.num_points + 1, dtype=np.intp)
        # starting_rank_arr = start_rank*np.ones(self.num_points, dtype=np.intp)

        starting_value = (<np.double_t *> starting_value_arr.data)
        # starting_rank = (<np.intp_t *> starting_rank_arr.data)

        cdef np.ndarray[np.double_t, ndim=2] knn_dist
        cdef np.ndarray[np.intp_t, ndim=2] knn_indices

        knn_dist, knn_indices = self.dist_tree.query(
                    self.tree.data,
                    k=self.max_neighbors_search + 1,
                    dualtree=True,
                    breadth_first=True,
                    )

#### Initialization of pure reciprocity then ranks are less than 2
        i = self.num_points
        while i:
            i -= 1
            j = self._pure_reciprocity(i, knn_indices, knn_dist)
            if j >= 0:
                self.U.reciprocal_emergence_of_energy_of_commonalities(i, j)
                self.result_add_edge(i, j)
#### initialization of reciprocities
            opt, opt_edge, opt_rank, opt_min = self._evaluate_reciprocity(i, knn_indices, knn_dist, start_rank)
            starting_value[i] = opt_min
        starting_value[self.num_points] = INF

        if self.result_edges >= self.num_points - 1:
            print ('Two subjects only')
            return

        global_opt, global_i, global_edge, global_rank = INF, -1, -1, 1
        while self.result_edges < self.num_points - 1:
            global_opt, global_i, global_edge, global_rank = INF, -1, -1, 1
            sorted_value_arr = np.argsort(starting_value_arr)
            sorteds = (<np.intp_t *> sorted_value_arr.data)

            s = 0
            i = sorteds[s]
            while starting_value[i] < global_opt:
                opt, opt_edge, opt_rank, opt_min = self._evaluate_reciprocity(i, knn_indices, knn_dist, start_rank)
                starting_value[i] = opt_min
                if opt < global_opt:
                    global_i, global_opt, global_edge, global_rank = i, opt, opt_edge, opt_rank
                s += 1
                i = sorteds[s]

            if global_opt == INF:
                print (str(self.num_points - 1 - self.result_edges) +' not connected edges. It is a forest. Try increasing max_neighbors(max_ranking) value '+str(self.max_neighbors_search)+' for a better result.')
                break

            cluster_arr = self.U.reciprocal_emergence_of_energy_of_commonalities(global_i, global_edge)
            self.result_add_edge(global_i, global_edge)
            for p in cluster_arr:
                self.result_add_cluster(p)
            # print (len(cluster_arr), self.result_edges, self.result_clusters, '======edges======', global_opt, 'i,j', global_i, global_edge, opt_rank, opt_min)

        self.result_clustered_energy = self.U.get_last_energy()
        # self.U.universal_completeness()
