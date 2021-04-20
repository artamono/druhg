# cython: language_level=3
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

# Builds minimum spanning tree for druhg algorithm
# uses dialectics to evaluate reciprocity
# Author: Pavel "DRUHG" Artamonov
# License: 3-clause BSD


import numpy as np
cimport numpy as np
import sys

import _heapq as heapq

from libc.stdlib cimport malloc, free

# cdef np.double_t INF = np.inf
cdef np.double_t INF = sys.float_info.max
# cdef np.double_t EPS = sys.float_info.min


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
    cdef:
        np.ndarray parent_arr
        np.intp_t *parent

        np.intp_t next_label

    def __init__(self, np.intp_t N):
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

    cdef:
        object tree
        object dist_tree

        np.double_t PRECISION

        np.intp_t num_points
        np.intp_t num_features

        np.intp_t max_neighbors_search

        UnionFind U
        np.intp_t result_edges
        np.ndarray result_value_arr
        np.ndarray result_pairs_arr

    def __init__(self, algorithm, tree, max_neighbors_search=16, metric='euclidean', leaf_size=20, is_slow = 0, double_precision = 0.000001, **kwargs):

        self.PRECISION = double_precision

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

    cdef void result_add_edge(self, np.intp_t a, np.intp_t b, Relation* rel):
        cdef np.intp_t i

        i = self.result_edges
        self.result_edges += 1

        self.result_pairs_arr[2*i] = a
        self.result_pairs_arr[2*i + 1] = b
        self.result_value_arr[i] = pow(rel.rec_dis, 2.)*(rel.rec_rank + rel.penalty)

        # Не понятно нужна ли мера на следующей стадии? С ней и без неё результаты почти сходятся.
        # Скорее всего она исчезает только после появления кластера. То есть для _druhg_amalgamation.border_overcoming будут две g, до и после. Тащить отдельно нет желания.
        # self.result_value_arr[i] *= pow(1.*rel.my_members/rel.rec_members, 0.5)

        # print(self.result_value_arr[i], self.result_edges, a,b, rel.rec_dis, rel.rec_rank + rel.penalty, rel.my_members, rel.rec_members)


    cdef bint _pure_reciprocity(self, np.intp_t i, np.ndarray[np.intp_t, ndim=2] knn_indices, np.ndarray[np.double_t, ndim=2] knn_dist, Relation* rel):
        """Finding pure reciprocal pairs(both ranks = 2)
        And deals with equal objects.
        Runs as initialization, short version of evaluate_reciprocity.
        Fixes problems when amount of same objects are less than K neighbors.
    
        Parameters
        ----------
    
        i : int
            index of the subject
    
        knn_indices: ndarray, shape (n_samples, n_features, )
            Array of arrays. Indices of first K neighbors(including itself, meaning i).
        
        knn_dist: ndarray, shape (n_samples, n_features, )
            Array of arrays. Distances of first K neighbors(including itself, meaning zero).
        
        rel: Relation
            Part of the output. Stores all significant parameters.            
            rel.reciprocity is zero if values are equal - it will lead to relaunch.
            rel.reciprocity is slightly different than in `evaluate_reciprocity`
            
        Returns
        -------
        res : bint
            Success if not zero.
        """
        cdef:
            np.intp_t ranki, j, \
                rank_left, rank_right, \
                parent

            np.double_t dis

        parent = self.U.fast_find(i)
        indices, distances = knn_indices[i], knn_dist[i]
        for ranki in range(0, self.max_neighbors_search + 1):
            j = indices[ranki]
            if parent == self.U.fast_find(j):
                continue

            dis = distances[ranki]
            if dis == 0.: # degenerate case.
                rel.reciprocity = 0.
                rel.target = j
                rel.my_rank = ranki
                rel.rec_rank = ranki
                rel.my_dis = 0.
                rel.rec_dis = 0.
                rel.penalty = 0
                rel.my_members = ranki
                rel.rec_members = 1

                return ranki + 1

            rank_left = bisect.bisect(distances, dis + self.PRECISION)
            if rank_left > 2:
                return 0

            rank_right = bisect.bisect(knn_dist[j], dis + self.PRECISION)
            if rank_right > 2:
                return 0

            rel.reciprocity = pow(dis,4)*4.
            rel.target = j
            rel.my_rank = 2
            rel.rec_rank = 2
            rel.my_dis = dis
            rel.rec_dis = dis
            rel.penalty = 0
            rel.my_members = 1
            rel.rec_members = 1

            return 2
        return 0

    cdef bint _evaluate_reciprocity(self, np.intp_t i, np.ndarray[np.intp_t, ndim=2] knn_indices, np.ndarray[np.double_t, ndim=2] knn_dist, Relation *rel):

        cdef:
            np.intp_t ranki, j, \
                p, parent, parent_opp, \
                members, opp_members, equal_members, \
                rank_left, rank_right, \
                opp_is_reachable, penalty, \
                res = 0

            np.double_t best, opt_min, \
                order, order_min, \
                dis, rank_dis, old_dis

            np.ndarray indices, ind_opp
            np.ndarray distances, dis_opp

        parent = self.U.fast_find(i)
        indices, distances = knn_indices[i], knn_dist[i]

        best, opt_min = INF, INF

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
            rank_right = bisect.bisect(dis_opp, dis + self.PRECISION) # reminder that bisect.bisect(dis_opp, dis) >= bisect.bisect_left(dis_opp, dis)
            if ranki > rank_right:
                continue
            rank_left = bisect.bisect(distances, dis + self.PRECISION)
            if rank_left > rank_right:
                continue

            rank_dis = distances[rank_right - 1]

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

            if rank_right == opp_is_reachable: # For upper bound. Adding 1 to fix rare case
                opp_is_reachable -= 1

            # val1 = rank_dis # [качество] без этого не отличить углов от ребер в квадрате. dis <= rank_dis <= 2*dis
            # val2 = rank_right + penalty # [количество] без этого не различить ядро квадрата от ребер. ranki <= rank_left <= rank_right
            # val3 = 1.*members/opp_members # [мера] без этого не обеспечить равномерное прирастание. 1/(rank_right - 1) < val3 < rank_right - 1

            order_min = pow(rank_dis, 4) * pow(rank_right + penalty, 2) * members # WARNING: `members` are counted improperly. If ranki != rank_left, then this pair might have same dist as the other. Real members might be higher.

            order = order_min / opp_members # WARNING: `opp_members` can be zero if knn_indices doesn't have itself. This rare case is possible when amount of equal objects less than K neighbors. `pure_reciprocity` initialization fixes this
            order_min = order_min / (rank_right - opp_is_reachable) # dividing by how many can be.
            if order_min < opt_min:
                opt_min = order_min
                rel.upper_bound = order_min # it will be used outside in the heap part

            if order < best:
                if res == 0:
                    res = 1
                best = order

                rel.reciprocity = best
                rel.target = j
                rel.my_rank = ranki
                rel.rec_rank = rank_right
                rel.my_dis = dis
                rel.rec_dis = rank_dis
                rel.penalty = penalty
                rel.my_members = members
                rel.rec_members = opp_members
                rel.upper_members = rank_right - opp_is_reachable

                # rel.value = pow(rel.rec_dis, 2) * (rel.rec_rank + rel.penalty) * pow(1.*rel.my_members/rel.rec_members, 0.5)

        return res


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

        cdef:
            np.intp_t i, k, \
                warn

            np.double_t upper_bound
            list restart, heap

            Relation rel, best_rel

            np.ndarray[np.double_t, ndim=2] knn_dist
            np.ndarray[np.intp_t, ndim=2] knn_indices

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
            if knn_dist[i][0] < 0.:
                print ('Distances cannot be negative! Exiting. ', i, knn_dist[i][0])
                return

            rel.reciprocity = 1.
            if self._pure_reciprocity(i, knn_indices, knn_dist, &rel):
                self.U.union(i, rel.target)
                self.result_add_edge(i, rel.target, &rel)

            if rel.reciprocity == 0.: # values match
                warn += 1
                i += 1 # need to relaunch same values
                continue

                # print ('pure', i,j)
#### initialization of reciprocities

            if self._evaluate_reciprocity(i, knn_indices, knn_dist, &rel):
                heapq.heappush(heap, (rel.upper_bound, i))

        if warn > 0:
            print ('A lot of values are the same. Try increasing self.max_neighbors_search.', warn)

        if self.result_edges >= self.num_points - 1:
            print ('Two subjects only')
            return

        while self.result_edges < self.num_points - 1:
            best_rel.reciprocity = INF
            del restart[:]

            k = len(heap)
            while k:
                k -= 1
                upper_bound, i = heapq.heappop(heap)
                if best_rel.reciprocity <= upper_bound:
                    restart.append((upper_bound, i))
                    break

                if self._evaluate_reciprocity(i, knn_indices, knn_dist, &rel):
                    restart.append((rel.upper_bound, i))
                    if rel.reciprocity + self.PRECISION < best_rel.reciprocity:
                        best_rel = rel
                        best_rel.index = i

            if best_rel.reciprocity == INF:
                print (str(self.num_points - 1 - self.result_edges) +' not connected edges. It is a forest. Try increasing max_neighbors(max_ranking) value '+str(self.max_neighbors_search)+' for a better result.')
                break

            self.U.union(best_rel.index, best_rel.target)
            self.result_add_edge(best_rel.index, best_rel.target, &best_rel)

            if self._evaluate_reciprocity(best_rel.index, knn_indices, knn_dist, &rel):
                heapq.heappush(heap, (rel.upper_bound, best_rel.index))

            for upper_bound, i in restart:
                if i != best_rel.index:
                    heapq.heappush(heap, (upper_bound, i))

    cdef _compute_tree_vicinity_heap(self):
        # DRUHG
        # computes DRUHG Spanning Tree
        # presumes that pop contains the best value
        # updates the vicinity of the newly added edge
        # stores and checks targets
        # faster than brute force and very accurate

        cdef:
            np.intp_t i, best_i, \
                p, p1, p2, \
                warn, has_data
            np.double_t upper_bound, best_value
            list discard, heap
            set _set, s
            dict amal_dic

            Relation rel

            np.ndarray[np.double_t, ndim=1] upper_arr
            np.double_t * upper

            np.ndarray[np.intp_t, ndim=1] target_arr
            np.intp_t * target

            np.ndarray[np.double_t, ndim=2] knn_dist
            np.ndarray[np.intp_t, ndim=2] knn_indices

        upper_arr = -1.*np.ones(self.num_points + 1, dtype=np.double)
        upper = (<np.double_t *> upper_arr.data)

        target_arr = -1*np.ones(self.num_points + 1, dtype=np.intp)
        target = (<np.intp_t *> target_arr.data)

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
            if knn_dist[i][0] < 0.:
                print ('Distances cannot be negative! Exiting. ', i, knn_dist[i][0])
                return

            upper[i] = INF
            rel.reciprocity = 1.
            if self._pure_reciprocity(i, knn_indices, knn_dist, &rel):
                self.U.union(i, rel.target)
                self.result_add_edge(i, rel.target, &rel)
                # print ('pure', i,j, value)
            if rel.reciprocity == 0.: # values match
                warn += 1
                i += 1 # need to relaunch same values
                continue

#### initialization of reciprocities
            if self._evaluate_reciprocity(i, knn_indices, knn_dist, &rel):
                upper[i] = rel.upper_bound
                heapq.heappush(heap, (rel.upper_bound, i))
                target[i] = rel.target

                p = self.U.fast_find(rel.target)
                if p not in amal_dic:
                    _ = set()
                    amal_dic[p] = _
                amal_dic[p].add(i)

        if warn > 0:
            print ('A lot of values are the same. Try increasing self.max_neighbors_search.', warn)

        if self.result_edges >= self.num_points - 1:
            print ('Two subjects only. Edges ', self.result_edges, '. Data points ', self.num_points - 1)
            return

        heapq.heappush(heap, (INF, self.num_points))

        _set = set()
        best_value = 0
        while best_value != INF:
            best_value, best_i = heapq.heappop(heap)
            if best_value != upper[best_i]:
                continue
            rel.target = 0
            rel.reciprocity = INF
            has_data = self._evaluate_reciprocity(best_i, knn_indices, knn_dist, &rel)

            if target[best_i] != rel.target:
                p = self.U.fast_find(target[best_i])
                if p in amal_dic:
                    _ = amal_dic[p]
                    _.discard(best_i)
                    amal_dic[p] = _

                p = self.U.fast_find(rel.target)
                if p not in amal_dic:
                    _ = set()
                    _.add(best_i)
                    amal_dic[p] = _
                else:
                    amal_dic[p].add(best_i)
                target[best_i] = rel.target

            if rel.reciprocity > best_value:
                upper[best_i] = rel.reciprocity
                if has_data:
                    if best_value < rel.upper_bound:
                        upper[best_i] = rel.upper_bound
                        heapq.heappush(heap, (rel.upper_bound, best_i))
                    else:
                        heapq.heappush(heap, (rel.reciprocity, best_i))
                continue
            best_value = rel.reciprocity

            p1 = self.U.fast_find(best_i)
            p2 = self.U.fast_find(rel.target)
            # adding edge
            self.U.union(best_i, rel.target)
            self.result_add_edge(best_i, rel.target, &rel)

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

                upper_bound = upper[i]
                if upper_bound != INF:
                    upper_bound /= 2
                    upper[i] = upper_bound
                    heapq.heappush(heap, (upper_bound, i))

            s.difference_update(discard)
            amal_dic[p] = s

            upper[best_i] = best_value
            heapq.heappush(heap, (best_value, best_i))

            # vicinity update
            # rank += 2
            # if rank > self.max_neighbors_search:
            #     rank = self.max_neighbors_search

            _set.clear()
            for k in range(0, rel.rec_rank):
                i = knn_indices[best_i][k]
                if self.U.fast_find(i) != p:
                    _set.add(i)
                i = knn_indices[rel.target][k]
                if self.U.fast_find(i) != p:
                    _set.add(i)

            for i in _set:
                upper_bound = upper[i]
                if upper_bound != INF:
                    upper_bound /= 2
                    upper[i] = upper_bound
                    heapq.heappush(heap, (upper_bound, i))

            if self.result_edges >= self.num_points - 1:
                break
