# cython: boundscheck=False
# cython: nonecheck=False
# Converts edge array into standard hierarchical clustering format
# Authors: Pavel "DRUHG" Artamonov
# License: 3-clause BSD

import numpy as np
cimport numpy as np

cdef class UnionFind (object):

    cdef np.ndarray parent_arr
    cdef np.ndarray num_points_arr
    cdef np.intp_t next_label
    cdef np.intp_t *parent
    cdef np.intp_t *num_points
    cdef np.intp_t count

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


cpdef np.ndarray[np.double_t, ndim=2] make_hierarchy(np.ndarray[np.double_t, ndim=1] L, np.ndarray[np.double_t, ndim=1] D):

    cdef np.ndarray[np.intp_t, ndim=1] sort_arr
    cdef np.ndarray[np.double_t, ndim=2] result_arr
    cdef np.double_t[:, ::1] result

    cdef np.intp_t N, aa, b, bb, index
    cdef np.double_t delta

    N = L.shape[0]

    result_arr = np.zeros((N - 1, 4))
    result = (<np.double_t[:N - 1, :4:1]> (
        <np.double_t *> result_arr.data))
    U = UnionFind(N)

    sort_arr = np.argsort(D)

    for index in range(N-1):

        a = sort_arr[index]
        b = <np.intp_t> L[a]

        delta = D[a]

        aa, bb = U.fast_find(a), U.fast_find(b)

        result[index][0] = aa
        result[index][1] = bb
        result[index][2] = delta
        result[index][3] = U.num_points[aa] + U.num_points[bb]

        U.union(aa, bb)

    return result_arr
