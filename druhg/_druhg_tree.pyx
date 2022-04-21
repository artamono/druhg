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
from libc.math cimport fabs, pow
import bisect

cdef np.double_t INF = sys.float_info.max

from sklearn.neighbors import KDTree, BallTree
# from sklearn import preprocessing
from joblib import Parallel, delayed

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

        np.intp_t n_jobs

        UnionFind U
        np.intp_t result_edges
        np.ndarray result_value_arr
        np.ndarray result_pairs_arr

        # np.ndarray result_extras1_arr # to extract everything else
        # np.ndarray result_extras2_arr # to extract everything else

    def __init__(self, algorithm, tree, max_neighbors_search=16, metric='euclidean', leaf_size=20, n_jobs=4, is_slow=0, **kwargs):

        self.PRECISION = kwargs.get('double_precision', 0.0000001) # this is only relevant if distances between datapoints are super small
        self.n_jobs = n_jobs

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

        # self.result_extras1_arr = np.empty(self.num_points - 1)
        # self.result_extras2_arr = np.empty(self.num_points - 1)

        self._compute_tree_edges(is_slow)

    cpdef tuple get_tree(self):
        return (self.result_pairs_arr[:self.result_edges*2].astype(int), self.result_value_arr[:self.result_edges])

    cpdef tuple get_extras(self):
        return (self.result_extras1_arr[:self.result_edges], self.result_extras2_arr[:self.result_edges])

    cdef void result_add_edge_debug(self, np.intp_t a, np.intp_t b, Relation* rel):
        cdef np.intp_t i

        i = self.result_edges
        self.result_edges += 1

        self.result_pairs_arr[2*i] = a
        self.result_pairs_arr[2*i + 1] = b
        self.result_value_arr[i] = rel.dia_dis
        # self.result_extras1_arr[i] = rel.rec_rank
        # self.result_extras2_arr[i] = rel.my_members

        # print(self.result_edges, '=================add===============', self.result_value_arr[i], ":", a,b, rel.reciprocity, '=', rel.rec_dis, rel.rec_rank, 'ranks', rel.my_rank, rel.rec_rank, rel.my_scope, 'm',rel.my_members, rel.rec_members)
        # print(rel.reciprocity, self.result_edges, a,b, rel.rec_dis, 'rR', rel.my_rank, rel.rec_rank, 'Dd', rel.rec_dis, rel.my_dis, 'Mm', rel.my_members, rel.rec_members)

    cdef void result_add_edge(self, np.intp_t a, np.intp_t b, np.double_t v):
        cdef np.intp_t i

        i = self.result_edges
        self.result_edges += 1

        self.result_pairs_arr[2*i] = a
        self.result_pairs_arr[2*i + 1] = b
        self.result_value_arr[i] = v


    cdef bint _pure_reciprocity(self, np.intp_t i, np.ndarray[np.intp_t, ndim=2] knn_indices, np.ndarray[np.double_t, ndim=2] knn_dist, Relation* rel, np.intp_t* infinitesimal):
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
            
        Infinitesimal: int
            Part of the output. Return distances smaller than the self.PRECISION level.
            Used to inform the user about hidden parameter.
            
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
                # rel.my_rank = ranki
                # rel.rec_rank = ranki
                # rel.my_dis = 0.
                rel.dia_dis = 0.
                # rel.my_members = ranki
                # rel.rec_members = 1
                # rel.index = i

                return ranki + 1

            if dis <= self.PRECISION:
                infinitesimal += 1

            rank_left = bisect.bisect(distances, dis + self.PRECISION)
            if rank_left > 2:
                return 0

            rank_right = bisect.bisect(knn_dist[j], dis + self.PRECISION)
            if rank_right > 2:
                return 0

            rel.reciprocity = dis
            rel.target = j
            # rel.my_rank = 2
            # rel.rec_rank = 2
            # rel.my_dis = dis
            rel.dia_dis = dis
            # rel.my_members = 1
            # rel.rec_members = 1
            # rel.index = i

            return 2
        return 0

    cdef bint _evaluate_reciprocity_slow(self, np.intp_t i, np.ndarray[np.intp_t, ndim=2] knn_indices, np.ndarray[np.double_t, ndim=2] knn_dist, Relation *rel):

        cdef:
            np.intp_t ranki, j, \
                parent, \
                rank_left, rank_right, \
                res = 0

            np.double_t best, order, \
                dis, rank_dis

            np.ndarray indices, ind_opp
            np.ndarray distances, dis_opp

        parent = self.U.fast_find(i)
        indices, distances = knn_indices[i], knn_dist[i]

        best = INF
        # print(self.max_neighbors_search, 'yo',distances)
        for ranki in range(0, self.max_neighbors_search + 1):
            j = indices[ranki]
            dis = distances[ranki]

            if parent == self.U.fast_find(j):
                continue

            if dis >= best + self.PRECISION:
                break

            dis_opp = knn_dist[j]
            rank_right = bisect.bisect(dis_opp, dis + self.PRECISION) - 1 # !reminder! bisect.bisect(dis_opp, dis) >= bisect.bisect_left(dis_opp, dis)
            if ranki > rank_right:
                continue

            rank_left = bisect.bisect(distances, dis + self.PRECISION) - 1
            if rank_left > rank_right:
                continue

            rank_dis = distances[rank_right]
            rank_left = len(set(indices[:rank_left+1]) - set(knn_indices[j][:rank_right+1])) + 1

            order = 1. * rank_right/rank_left * rank_dis

            # print(i, order < best, order, '=', rank_right, m, rank_dis, 'rl', rank_left, dis)
            # print(indices[:rank_left+1], knn_indices[j][:rank_right+1])
            # print(distances[:rank_left+1], knn_dist[j][:rank_right+1])
            # print(distances[:rank_left+2], knn_dist[j][:rank_right+2])

            if order < best: # minimizing
                res = 1
                best = order

                rel.reciprocity = best
                rel.target = j
                # rel.my_dis = dis
                rel.dia_dis = rank_dis
                # rel.loop_rank = ranki
                # rel.my_rank = rank_left
                # rel.rec_rank = rank_right
                # rel.my_scope = scope_left
                # rel.my_members = members
            # print (i,j, rel.reciprocity, ':', rel.my_dis, rel.rec_dis, ":", rel.my_rank, rel.rec_rank, rel.my_scope, ".", rel.my_members)
            #     print (i,j, rel.reciprocity, ':', rel.my_dis, rel.rec_dis, ":", rel.my_rank, rel.rec_rank, rel.my_scope, ".", rel.my_members)
                # rel.value = pow(rel.rec_dis, 2) * (rel.rec_rank) * pow(1.*rel.my_members/rel.rec_members, 0.5)

        return res

    cdef bint _evaluate_reciprocity_fast(self, np.intp_t i, np.ndarray[np.intp_t, ndim=2] knn_indices, np.ndarray[np.double_t, ndim=2] knn_dist, Relation *rel):

        cdef:
            np.intp_t ranki, j, \
                parent, \
                rank_left, rank_right, \
                res = 0

            np.double_t best, order, \
                dis, rank_dis

            np.ndarray indices, ind_opp
            np.ndarray distances, dis_opp

        parent = self.U.fast_find(i)
        indices, distances = knn_indices[i], knn_dist[i]

        best = INF
        # print(self.max_neighbors_search, 'yo',distances)
        for ranki in range(0, self.max_neighbors_search + 1):
            j = indices[ranki]
            dis = distances[ranki]

            if parent == self.U.fast_find(j):
                continue

            if dis >= best + self.PRECISION:
                break

            dis_opp = knn_dist[j]
            rank_right = bisect.bisect(dis_opp, dis + self.PRECISION) - 1 # !reminder! bisect.bisect(dis_opp, dis) >= bisect.bisect_left(dis_opp, dis)
            if ranki > rank_right:
                continue

            rank_left = bisect.bisect(distances, dis + self.PRECISION) - 1
            if rank_left > rank_right:
                continue

            rank_dis = distances[rank_right]
            # rank_left = 1.*len(set(indices[:rank_left+1]) - set(knn_indices[j][:rank_right+1])) + 1 # slow/proper
            order = 1. * rank_right/rank_left * rank_dis

            # print(i, order < best, order, '=', rank_right, m, rank_dis, 'rl', rank_left, dis)
            # print(indices[:rank_left+1], knn_indices[j][:rank_right+1])
            # print(distances[:rank_left+1], knn_dist[j][:rank_right+1])
            # print(distances[:rank_left+2], knn_dist[j][:rank_right+2])

            if order < best: # minimizing
                res = 1
                best = order

                rel.reciprocity = best
                rel.target = j
                # rel.my_dis = dis
                rel.dia_dis = rank_dis
                # rel.loop_rank = ranki
                # rel.my_rank = rank_left
                # rel.rec_rank = rank_right
                # rel.my_scope = scope_left
                # rel.my_members = members
            # print (i,j, rel.reciprocity, ':', rel.my_dis, rel.rec_dis, ":", rel.my_rank, rel.rec_rank, rel.my_scope, ".", rel.my_members)
            #     print (i,j, rel.reciprocity, ':', rel.my_dis, rel.rec_dis, ":", rel.my_rank, rel.rec_rank, rel.my_scope, ".", rel.my_members)
                # rel.value = pow(rel.rec_dis, 2) * (rel.rec_rank) * pow(1.*rel.my_members/rel.rec_members, 0.5)

        return res


    cdef _compute_tree_edges(self, is_slow):
        # DRUHG
        # computes DRUHG Spanning Tree
        # uses heap

        cdef:
            np.intp_t i,j, \
                warn, infinitesimal
            np.double_t v

            list heap
            Relation rel

            np.ndarray[np.double_t, ndim=2] knn_dist
            np.ndarray[np.intp_t, ndim=2] knn_indices

        if self.tree.data.shape[0] > 16384 and self.n_jobs > 1: # multicore 2-3x speed up for big datasets
            split_cnt = self.num_points // self.n_jobs
            datasets = []
            for i in range(self.n_jobs):
                if i == self.n_jobs - 1:
                    datasets.append(np.asarray(self.tree.data[i*split_cnt:]))
                else:
                    datasets.append(np.asarray(self.tree.data[i*split_cnt:(i+1)*split_cnt]))

            knn_data = Parallel(n_jobs=self.n_jobs)(
                delayed(self.tree.query)
                (points,
                 self.max_neighbors_search + 1,
                 dualtree=True,
                 breadth_first=True
                 )
                for points in datasets)
            knn_dist = np.vstack([x[0] for x in knn_data])
            knn_indices = np.vstack([x[1] for x in knn_data])
        else:
            knn_dist, knn_indices = self.dist_tree.query(
                        self.tree.data,
                        k=self.max_neighbors_search + 1,
                        dualtree=True,
                        breadth_first=True,
                        )

        warn, infinitesimal = 0, 0
        heap = []
#### Initialization of pure reciprocity then ranks are less than 2
        i = self.num_points
        while i:
            i -= 1
            if knn_dist[i][0] < 0.:
                print ('Distances cannot be negative! Exiting. ', i, knn_dist[i][0])
                return

            rel.reciprocity = 1.
            if self._pure_reciprocity(i, knn_indices, knn_dist, &rel, &infinitesimal):
                self.U.union(i, rel.target)
                self.result_add_edge(i, rel.target, rel.dia_dis)

            if rel.reciprocity == 0.: # values match
                warn += 1
                i += 1 # need to relaunch same values
                continue

                # print ('pure', i,j)
#### initialization of reciprocities
            if (is_slow and self._evaluate_reciprocity_slow(i, knn_indices, knn_dist, &rel))\
            or (not is_slow and self._evaluate_reciprocity_fast(i, knn_indices, knn_dist, &rel)):
                heapq.heappush(heap, (rel.reciprocity, i, rel.target, rel.dia_dis))
                # print('RESULT', rel.reciprocity, i)
        # return
        if warn > 0:
            print ('A lot of values(',warn,') are the same. Try increasing max_neighbors_search(',self.max_neighbors_search,') parameter.')

        if infinitesimal > 0:
            print ('Some distances(', infinitesimal, ') are smaller than self.PRECISION (', self.PRECISION,') level. Try decreasing double_precision parameter.')

        if self.result_edges >= self.num_points - 1:
            print ('Two subjects only')
            return

        heapq.heappush(heap, (INF, -1,-1,0.))
        if is_slow:
            while self.result_edges < self.num_points - 1:
                best, i, j, v = heapq.heappop(heap)

                if best == INF:
                    print (str(self.num_points - 1 - self.result_edges) +' not connected edges. It is a forest. Try increasing max_neighbors(max_ranking) value '+str(self.max_neighbors_search)+' for a better result.')
                    break

                if self.U.fast_find(i)!=self.U.fast_find(j):
                    self.U.union(i, j)
                    self.result_add_edge(i, j, v)
                if self._evaluate_reciprocity_slow(i, knn_indices, knn_dist, &rel):
                    heapq.heappush(heap, (rel.reciprocity, i, rel.target, rel.dia_dis)) # updated value is in
        else:
            while self.result_edges < self.num_points - 1:
                best, i, j, v = heapq.heappop(heap)

                if best == INF:
                    print (str(self.num_points - 1 - self.result_edges) +' not connected edges. It is a forest. Try increasing max_neighbors(max_ranking) value '+str(self.max_neighbors_search)+' for a better result.')
                    break

                if self.U.fast_find(i)!=self.U.fast_find(j):
                    self.U.union(i, j)
                    self.result_add_edge(i, j, v)
                if self._evaluate_reciprocity_fast(i, knn_indices, knn_dist, &rel):
                    heapq.heappush(heap, (rel.reciprocity, i, rel.target, rel.dia_dis)) # updated value is in

        return
        # less storage, reevaluating to get the data
        # heapq.heappush(heap, (INF, -1))
        # if is_slow:
        #     while self.result_edges < self.num_points - 1:
        #         best, i = heapq.heappop(heap)
        #         if best == INF:
        #             print (str(self.num_points - 1 - self.result_edges) +' not connected edges. It is a forest. Try increasing max_neighbors(max_ranking) value '+str(self.max_neighbors_search)+' for a better result.')
        #             break
        #         if self._evaluate_reciprocity_slow(i, knn_indices, knn_dist, &rel):
        #             if rel.reciprocity <= best + self.PRECISION:
        #                 self.U.union(i, rel.target)
        #                 self.result_add_edge(i, rel.target, &rel)
        #                 if self._evaluate_reciprocity_slow(i, knn_indices, knn_dist, &rel):
        #                     heapq.heappush(heap, (rel.reciprocity, i)) # updated value is in
        #             else: # the result got worse. When equal values are connected.
        #                 heapq.heappush(heap, (rel.reciprocity, i))
        # else:
        #     while self.result_edges < self.num_points - 1:
        #         best, i = heapq.heappop(heap)
        #         if best == INF:
        #             print (str(self.num_points - 1 - self.result_edges) +' not connected edges. It is a forest. Try increasing max_neighbors(max_ranking) value '+str(self.max_neighbors_search)+' for a better result.')
        #             break
        #         # по идее можем присоединять? но тогда нужно тащить target и diadis
        #         if self._evaluate_reciprocity_fast(i, knn_indices, knn_dist, &rel):
        #             if rel.reciprocity <= best + self.PRECISION:
        #                 self.U.union(i, rel.target)
        #                 self.result_add_edge(i, rel.target, &rel)
        #                 if self._evaluate_reciprocity_fast(i, knn_indices, knn_dist, &rel):
        #                     heapq.heappush(heap, (rel.reciprocity, i)) # updated value is in
        #             else: # the result got worse. When equal values are connected.
        #                 heapq.heappush(heap, (rel.reciprocity, i))
