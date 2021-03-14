# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# builds minimum spanning tree for druhg algorithm
# Author: Pavel "DRUHG" Artamonov
# License: 3-clause BSD


import numpy as np
cimport numpy as np
import sys

import _heapq as heapq

from libc.stdlib cimport malloc, free

cdef np.double_t INF = np.inf
cdef np.double_t EPS = sys.float_info.min

from libc.math cimport fabs, pow

from sklearn.neighbors import KDTree, BallTree
from sklearn import preprocessing

import bisect

# from sklearn.externals.joblib import Parallel, delayed

cdef class PairwiseDistanceTreeSparse(object):
    cdef object data_arr
    cdef int data_size

    def __init__(self, N, d):
        self.data_size = N
        self.data_arr = d

    cpdef tuple query(self, d, k, dualtree = 0, breadth_first = 0):
        # TODO: actually we need to consider replacing INF with something else.
        # Reciprocity of absent link is not the same as the INF. Do reciprocity with graphs!
        cdef np.ndarray[np.double_t, ndim=2] knn_dist
        cdef np.ndarray[np.intp_t, ndim=2] knn_indices

        knn_dist = INF*np.ones((self.data_size, k+1))
        knn_indices = np.zeros((self.data_size, k+1), dtype=np.intp)

        warning = 0

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
            else:
                # edge loops itself
                warning += 1
                while j:
                    j -= 1
                    knn_dist[i][j] = data[sorted[j]]
                    knn_indices[i][j] = idx[sorted[j]]

            knn_dist[i][0], knn_indices[i][0] = 0., i # have to add itself. Edge to itself have to be zero!

        if warning:
            print ('Attention!: Sparse matrix has an edge that forms a loop! They were zeroed.', warning)

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

cdef class UnionFind (object):
    cdef np.ndarray parent_arr
    cdef np.intp_t *parent

    cdef np.intp_t next_label

    def __init__(self, N):
        self.parent_arr = np.zeros(2 * N, dtype=np.intp)
        self.parent = (<np.intp_t *> self.parent_arr.data)
        self.next_label = N + 1

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

    # cdef np.intp_t is_cluster(self, np.intp_t n):
    #     return self.parent[n]

    cdef void union(self, np.intp_t aa, np.intp_t bb):
        aa, bb = self.fast_find(aa), self.fast_find(bb)

        self.parent[aa] = self.parent[bb] = self.next_label
        self.next_label += 1
        return

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

    cdef UnionFind U
    cdef np.intp_t result_edges
    cdef np.ndarray result_value_arr
    cdef np.ndarray result_pairs_arr

    def __init__(self, algorithm, tree, max_neighbors_search=16, metric='euclidean', leaf_size=20, is_slow = 0, **kwargs):

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

        self.U = UnionFind(self.num_points)

        self.result_edges = 0

        self.result_pairs_arr = np.empty((self.num_points*2 - 2))
        self.result_value_arr = np.empty(self.num_points - 1)

        self._compute_tree_edges(is_slow)

    cpdef tuple get_tree(self):
        return (self.result_pairs_arr[:self.result_edges*2].astype(int), self.result_value_arr[:self.result_edges])

    cdef void result_add_edge(self, np.intp_t a, np.intp_t b, np.double_t val):
        cdef np.intp_t i

        i = self.result_edges
        self.result_pairs_arr[2*i] = a
        self.result_pairs_arr[2*i + 1] = b
        self.result_value_arr[i] = val
        self.result_edges += 1

    cdef tuple _pure_reciprocity(self, i, knn_indices, knn_dist):
        cdef np.intp_t ranki, j, \
            rank_left, rank_right, \
            parent

        cdef np.double_t dis

        parent = self.U.fast_find(i)
        indices, distances = knn_indices[i], knn_dist[i]
        for ranki in range(0, 2):
            j = indices[ranki]
            if parent == self.U.fast_find(j):
                continue

            dis = distances[ranki]
            if dis == 0.: # degenerate case. If all values are equal it will blow up in _evaluate_reciprocity
                return j, 0.

            rank_left = bisect.bisect(distances, dis)
            if rank_left > 2:
                return -1, -1.

            rank_right = bisect.bisect(knn_dist[j], dis)
            if rank_right > 2:
                return -1, -1.
            return j, dis
        return -1, -1.

    cdef tuple _evaluate_reciprocity(self, i, knn_indices, knn_dist):

        cdef np.intp_t ranki, j, opt_edge, \
            p, parent, parent_opp, \
            members, opp_members, equal_members, \
            rank_left, rank_right, scope_size, opt_rank, \
            opp_is_reachable, penalty

        cdef np.double_t best, opt_min, \
            order, order_min, \
            dis, rank_dis, old_dis, \
            val1, val2

        parent = self.U.fast_find(i)
        indices, distances = knn_indices[i], knn_dist[i]

        best, opt_min = INF, INF
        opt_rank = 0
        opt_edge = 0
        val1, val2 = 1., 1.
        members = 0
        equal_members = 0
        old_dis = 0.

        # opt_dump = (i)
        for ranki in range(0, self.max_neighbors_search + 1):
            j = indices[ranki]
            dis = distances[ranki]

            if dis != old_dis:
                members += equal_members
                old_dis = dis
                equal_members = 0

            if parent == self.U.fast_find(j):
                equal_members += 1
                continue

            # if pow(dis,4) * ranki**2 * members >= best * (ranki - 1):
            if pow(dis,4) * ranki * members >= best:
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

            opp_members = 0
            opp_is_reachable = 0
            for s in ind_opp[:rank_right]:
                p = self.U.fast_find(s)
                opp_members += parent_opp==p
                opp_is_reachable += parent==p

            penalty = 0 # penalizing in case of reaching the limit of max_neighbors_search
            if opp_is_reachable == 0: # rank_right >= self.max_neighbors_search:
                penalty = rank_left

            # todo: create a test to reveal this
            # if opp_members == 0: # this can happen if all opposing values are equal
            #     opp_members = 1
            #     print ('opp_members', opp_members)
            #
            # if rank_right - opp_is_reachable == 0:
            #     rank_right += 1
            #     print ('omg', rank_right, opp_is_reachable)

            val1 = rank_dis # [качество] без этого не отличить углов от ребер в квадрате. dis <= rank_dis <= 2*dis
            val2 = rank_right + penalty # [количество] без этого не различить ядро квадрата от ребер. ranki <= rank_left <= rank_right = scope_size
            # val3 = 1.*members/opp_members # [мера] без этого не обеспечить равномерное прирастание. 1/scope_size < val3 < scope_size

            order_min = pow(val1, 4) * pow(val2, 2) * members
            order = order_min / opp_members
            order_min = order_min / (rank_right - opp_is_reachable)

            if order_min < opt_min:
                opt_min = order_min

            if order < best:
                best, opt_edge = order, j
                opt_rank = rank_right
                # opt_dump = (i, j, order, dis, rank_dis, ranki, rank_left, rank_right, penalty, members, opp_members)

            # if i == 6918: # or j == 6998:
            #     f1 = lambda x: self.U.fast_find(x)==parent_opp
            #     f2 = lambda x: self.U.fast_find(x)==parent

                # print (i, j, 'vals', val1, val2, 'mems', members, opp_members, 'ranks', ranki, rank_left, rank_right, 'order', order, order_min, 'opts', opt, opt_min, opt_edge)
                # print ('ind', ind_opp[:rank_right], dis_opp[:rank_right], [ f1(x) for x in ind_opp[:rank_right]], [ f2(x) for x in ind_opp[:rank_right]], indices[:ranki], distances[:ranki])
            # print (i, j, 'vals', val1, val2, val3, 'mems', members, opp_members, 'order', order, 'opts', opt, opt_edge, opt_rank, opt_min)
        # print ('opt_dump', opt_dump)
        return best, opt_edge, opt_min, opt_rank


    cdef _compute_tree_edges(self, is_slow):
        # if algorithm == 'deterministic' or algorithm == 'slow':
        if is_slow:
            # almost a brute force
            self._compute_tree_deterministic_heap()
        else:
            self._compute_tree_vicinity_heap()

    cdef _compute_tree_deterministic_heap(self):
        # DRUHG
        # computes DRUHG Spanning Tree
        # uses heap and near brute force

        cdef np.intp_t i, j, \
            best_i, best_j, rank, \
            warn
        cdef np.double_t value, opt_min, best_value
        cdef list cluster_arr, restart, heap

        cdef np.ndarray[np.double_t, ndim=2] knn_dist
        cdef np.ndarray[np.intp_t, ndim=2] knn_indices

        knn_dist, knn_indices = self.dist_tree.query(
                    self.tree.data,
                    k=self.max_neighbors_search + 1,
                    dualtree=True,
                    breadth_first=True,
                    )
        warn = 0
        heap, restart = [], []
#### Initialization of pure reciprocity then ranks are less than 2
        i = self.num_points
        while i:
            i -= 1
            j, value = self._pure_reciprocity(i, knn_indices, knn_dist)
            if j >= 0:
                value = pow(value,2)*2.
                self.U.union(i, j)
                self.result_add_edge(i, j, value)

            if value == 0. and knn_dist[i][self.max_neighbors_search - 1] == 0.: # all values are equal. We skip or bust
                warn += 1
                continue

                # print ('pure', i,j)
#### initialization of reciprocities
            value, j, opt_min, rank = self._evaluate_reciprocity(i, knn_indices, knn_dist)
            if value != INF:
                heapq.heappush(heap, (opt_min, i))
        heapq.heappush(heap, (INF, self.num_points))

        if warn > 0:
            print ('A lot of values are the same. Cases: '+str(warn)+'. Try increasing self.max_neighbors_search: '+str(self.max_neighbors_search) )

        if self.result_edges >= self.num_points - 1:
            print ('Two subjects only')
            return

        best_i, best_j = 0, 0
        while self.result_edges < self.num_points - 1:
            best_value = INF
            del restart[:]
            while True:
                value, i = heapq.heappop(heap)
                if best_value <= value:
                    if value != INF or i == self.num_points:
                        restart.append((value, i))
                    break
                value, j, opt_min, rank = self._evaluate_reciprocity(i, knn_indices, knn_dist)

                restart.append((opt_min, i))
                if value < best_value:
                    best_i, best_value, best_j = i, value, j

            if best_value == INF:
                print (str(self.num_points - 1 - self.result_edges) +' not connected edges. It is a forest. Try increasing max_neighbors(max_ranking) value '+str(self.max_neighbors_search)+' for a better result.')
                break

            best_value = pow(best_value,0.5)
            self.U.union(best_i, best_j)
            self.result_add_edge(best_i, best_j, best_value)

            value, best_j, opt_min, rank = self._evaluate_reciprocity(best_i, knn_indices, knn_dist)
            if value != INF:
                heapq.heappush(heap, (opt_min, best_i))

            for value, i in restart:
                if i != best_i and value != INF:
                    heapq.heappush(heap, (value, i))
                if i == self.num_points:
                    heapq.heappush(heap, (INF, self.num_points))


    cdef _compute_tree_vicinity_heap(self):
        # DRUHG
        # computes DRUHG Spanning Tree
        # presumes that pop contains the best value
        # updates the vicinity of the newly added edge
        # stores and checks targets
        # faster than brute force and very accurate

        cdef np.intp_t i, j, \
            best_i, best_j, rank, \
            p, p1, p2, \
            warn
        cdef np.double_t value, opt_min, best_value
        cdef list cluster_arr, discard, heap
        cdef set _set, s

        cdef np.ndarray[np.double_t, ndim=1] start_arr
        cdef np.double_t * start
        start_arr = -1.*np.ones(self.num_points + 1, dtype=np.intp)
        start = (<np.double_t *> start_arr.data)

        cdef np.ndarray[np.intp_t, ndim=1] target_arr
        cdef np.intp_t * target
        target_arr = -1*np.ones(self.num_points + 1, dtype=np.intp)
        target = (<np.intp_t *> target_arr.data)

        cdef np.ndarray[np.double_t, ndim=2] knn_dist
        cdef np.ndarray[np.intp_t, ndim=2] knn_indices

        knn_dist, knn_indices = self.dist_tree.query(
                    self.tree.data,
                    k=self.max_neighbors_search + 1,
                    dualtree=True,
                    breadth_first=True,
                    )
        warn = 0
        heap, discard = [], []
        amal_dic = {}
#### Initialization of pure reciprocity then ranks are less than 2
        i = self.num_points
        while i:
            i -= 1
            j, value = self._pure_reciprocity(i, knn_indices, knn_dist)
            if j >= 0:
                value = value = pow(value,2)*2.
                self.U.union(i, j)
                self.result_add_edge(i, j, value)
                # print ('pure', i,j, value)
            if value == 0. and knn_dist[i][self.max_neighbors_search - 1] == 0.: # all values are equal. We skip or bust
                warn += 1
                continue

#### initialization of reciprocities
            value, j, opt_min, rank = self._evaluate_reciprocity(i, knn_indices, knn_dist)
            start[i] = opt_min
            if value != INF:
                heapq.heappush(heap, (opt_min, i))
                target[i] = j

                p = self.U.fast_find(j)
                if p not in amal_dic:
                    _ = set()
                    amal_dic[p] = _
                amal_dic[p].add(i)

        heapq.heappush(heap, (INF, self.num_points))

        if warn > 0:
            print ('A lot of values are the same. Cases: '+str(warn)+'. Try increasing self.max_neighbors_search: '+str(self.max_neighbors_search) )

        if self.result_edges >= self.num_points - 1:
            print ('Two subjects only. Edges ', self.result_edges, '. Data points ', self.num_points - 1)
            return

        _set = set()
        best_value = 0
        while best_value != INF and self.result_edges < self.num_points - 1:
            best_value, best_i = heapq.heappop(heap)
            value = start[best_i]
            if best_value != value:
                continue
            value, best_j, opt_min, rank = self._evaluate_reciprocity(best_i, knn_indices, knn_dist)

            if target[best_i] != best_j:
                p = self.U.fast_find(target[best_i])
                if p in amal_dic:
                    _ = amal_dic[p]
                    _.discard(best_i)
                    amal_dic[p] = _

                p = self.U.fast_find(best_j)
                if p not in amal_dic:
                    _ = set()
                    _.add(best_i)
                    amal_dic[p] = _
                else:
                    amal_dic[p].add(best_i)
                target[best_i] = best_j

            if value > best_value:
                if value != INF:
                    if best_value < opt_min:
                        value = opt_min
                    heapq.heappush(heap, (value, best_i))
                    # print ('rerun!', best_i, best_j, value, best_value, opt_min)
                start[best_i] = value
                continue
            best_value = value

            p1 = self.U.fast_find(best_i)
            p2 = self.U.fast_find(best_j)
            # adding edge
            value = pow(value,0.5)
            self.U.union(best_i, best_j)
            self.result_add_edge(best_i, best_j, value)

            # if len(cluster_arr)>0 or 1:
            #     print ('clusters', self.result_clusters, '(+', len(cluster_arr), ') edges', self.result_edges,  '======opt======', value, 'i,j', best_i, best_j, opt_dump)
            #
            #     print ('  ============', dump_com[0])
            #     print ('    === ', dump_com[1])
            #     print ('    === ', dump_com[2])
            #     print ('  ===== ', dump_com[3])
            #     print ('heap_vicinity3', value, 'i,j', best_i, best_j, dump_com, opt_dump)

            # update of all who targeted new amalgamation
            p = self.U.fast_find(best_i) # after union
            if p1 in amal_dic:
                s = amal_dic[p1]
            else:
                s = set()
            if p2 in amal_dic:
                s.update(amal_dic[p2])

            s.discard(best_i)

            del discard[:]
            for i in s: # everyone in here is targeting new amalgamation
                if p == self.U.fast_find(i):
                    discard.append(i)
                    target[i] = i
                    continue

                value = start[i]
                if value != INF:
                    value /= 2
                    start[i] = value
                    heapq.heappush(heap, (value, i))

            s.difference_update(discard)
            amal_dic[p] = s

            start[best_i] = best_value
            heapq.heappush(heap, (best_value, best_i))

            # vicinity update
            # rank += 2
            # if rank > self.max_neighbors_search:
            #     rank = self.max_neighbors_search

            _set.clear()
            for k in range(0, rank):
                i = knn_indices[best_i][k]
                if self.U.fast_find(i) != p:
                    _set.add(i)
                i = knn_indices[best_j][k]
                if self.U.fast_find(i) != p:
                    _set.add(i)

            for i in _set:
                value = start[i]
                if value != INF:
                    value /= 2
                    start[i] = value
                    heapq.heappush(heap, (value, i))
