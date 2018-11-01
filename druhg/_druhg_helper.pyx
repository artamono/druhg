# cython: boundscheck=False
# cython: nonecheck=False
# Converts edge list into standard hierarchical clustering format
# Authors: Leland McInnes, Steve Astels
# License: 3-clause BSD

import numpy as np
cimport numpy as np

cdef class UnionFind (object):

    cdef np.ndarray parent_arr
    cdef np.ndarray num_points_arr
    cdef np.intp_t next_label
    cdef np.intp_t *parent
    cdef np.intp_t *num_points

    def __init__(self, N):
        self.parent_arr = -1 * np.ones(2 * N - 1, dtype=np.intp, order='C')
        self.next_label = N
        self.num_points_arr = np.hstack((np.ones(N, dtype=np.intp),
                                   np.zeros(N-1, dtype=np.intp)))
        self.parent = (<np.intp_t *> self.parent_arr.data)
        self.num_points = (<np.intp_t *> self.num_points_arr.data)

    cdef void union(self, np.intp_t m, np.intp_t n):
        self.num_points[self.next_label] = self.num_points[m] + self.num_points[n]
        self.parent[m] = self.next_label
        self.parent[n] = self.next_label
        self.num_points[self.next_label] = self.num_points[m] + self.num_points[n]
        self.next_label += 1

        return

    cdef np.intp_t fast_find(self, np.intp_t n):
        cdef np.intp_t p
        p = n
        while self.parent_arr[n] != -1:
            n = self.parent_arr[n]
        # label up to the root
        while self.parent_arr[p] != n:
            p, self.parent_arr[p] = self.parent_arr[p], n
        return n


cpdef np.ndarray[np.double_t, ndim=2] make_hierarchy(np.ndarray[np.double_t, ndim=2] L):

    cdef np.ndarray[np.double_t, ndim=2] result_arr
    cdef np.double_t[:, ::1] result

    cdef np.intp_t N, a, aa, b, bb, index
    cdef np.double_t delta

    result_arr = np.zeros((L.shape[0], L.shape[1] + 1))
    result = (<np.double_t[:L.shape[0], :4:1]> (
        <np.double_t *> result_arr.data))
    N = L.shape[0] + 1
    U = UnionFind(N)

    for index in range(L.shape[0]):

        a = <np.intp_t> L[index, 0]
        b = <np.intp_t> L[index, 1]
        delta = L[index, 2]

        aa, bb = U.fast_find(a), U.fast_find(b)

        result[index][0] = aa
        result[index][1] = bb
        result[index][2] = delta
        result[index][3] = U.num_points[aa] + U.num_points[bb]

        U.union(aa, bb)

    return result_arr
