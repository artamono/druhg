# cython: boundscheck=False
# cython: nonecheck=False
# cython: initializedcheck=False
# DRUHG
# even rankability distance computations
# Authors: Pavel "DRUHG" Artamonov
# License: 3-clause BSD

import numpy as np
cimport numpy as np

import sys

from scipy.sparse import lil_matrix as sparse_matrix
import gc


def even_rankability(distance_matrix, min_flatting=0):
    """Compute the weighted adjacency matrix of the mutual rankability
    graph of a distance matrix.

    Parameters
    ----------
    distance_matrix : ndarray, shape (n_samples, n_samples)
        Array of distances between samples.

    min_flatting : nint, optional (default=0)
        The min_flatting paramater of DRUHG - how many neighbors of the point to flat rank and even their distances


    Returns
    -------
    even_rankability: ndarray, shape (n_samples, n_samples)
        Weighted adjacency matrix of the mutual rankability graph.

    pushed_ranks: ndarray, shape (n_samples)
        Vector of cummulated pushed ranks

    ranks_equalized: int
        Total amount of ranks changed during comparison

    """

    assert(distance_matrix.shape[0] == distance_matrix.shape[1])

    size = distance_matrix.shape[0]

    if min_flatting>size:
        min_flatting = size # this will give a funny result )))

    result = np.copy(distance_matrix)
    net_ranks = 0

# sorting is not working properly because of double to float cython convertion, argsort is ok
    sortedM = np.argsort(result, axis=1)

    for i in range(0, size-1):
        for j in range(i+1, size):
            distance = distance_matrix[i][j]

            # rank_i = len(distance_matrix[distance_matrix[i]<distance])
            # rank_j = len(distance_matrix[distance_matrix[j]<distance])

            rank_i = np.where(sortedM[i]==j)[0][0]
            rank_j = np.where(sortedM[j]==i)[0][0]

            while (distance_matrix[i][sortedM[i][rank_i - 1]]==distance):
                rank_i -= 1

            while (distance_matrix[j][sortedM[j][rank_j - 1]]==distance):
                rank_j -= 1

            if rank_i > rank_j:
                rank = rank_i

                if rank < min_flatting:
                    rank = min_flatting
                if (distance != distance_matrix[j][sortedM[j][rank]]):
                    result[i][j] = result[j][i] = distance_matrix[j][sortedM[j][rank]]
                    rr = 0
                    rank_j += 1
                    while (rank_j <= rank):
                        if (distance != distance_matrix[j][sortedM[j][rank_j]]):
                            rr += 1
                        rank_j += 1
                    # pushed_ranks[i] += rr
                    # pusher_ranks[j] += rr
                    net_ranks += rr
            else:
                rank = rank_j

                if rank < min_flatting:
                    rank = min_flatting
                if (distance != distance_matrix[i][sortedM[i][rank]]):
                    result[i][j] = result[j][i] = distance_matrix[i][sortedM[i][rank]]
                    rr = 0
                    rank_i += 1
                    while (rank_i <= rank):
                        if (distance != distance_matrix[i][sortedM[i][rank_i]]):
                            rr += 1
                        rank_i += 1
                    # pushed_ranks[j] += rr
                    # pusher_ranks[i] += rr
                    net_ranks += rr

            # distance = np.max([distance_matrix[i][sortedM[i][rank]],distance_matrix[j][sortedM[j][rank]]])

    return result, net_ranks#, pushed_ranks, pusher_ranks,


# cpdef sparse_even_rankability(object lil_matrix, np.intp_t min_points=5,
#                                  float alpha=1.0):
# 
#     cdef np.intp_t i
#     cdef np.intp_t j
#     cdef np.intp_t n
#     cdef np.double_t mr_dist
#     cdef list sorted_row_data
#     cdef np.ndarray[dtype=np.double_t, ndim=1] core_distance
#     cdef np.ndarray[dtype=np.int32_t, ndim=1] nz_row_data
#     cdef np.ndarray[dtype=np.int32_t, ndim=1] nz_col_data
# 
#     result = sparse_matrix(lil_matrix.shape)
#     core_distance = np.empty(lil_matrix.shape[0], dtype=np.double)
# 
#     for i in range(lil_matrix.shape[0]):
#         sorted_row_data = sorted(lil_matrix.data[i])
#         if min_points < len(sorted_row_data):
#             core_distance[i] = sorted_row_data[min_points]
#         else:
#             core_distance[i] = np.infty
# 
#     nz_row_data, nz_col_data = lil_matrix.nonzero()
# 
#     for n in range(nz_row_data.shape[0]):
#         i = nz_row_data[n]
#         j = nz_col_data[n]
# 
#         mr_dist = max(core_distance[i], core_distance[j], lil_matrix[i, j])
#         if np.isfinite(mr_dist):
#             result[i, j] = mr_dist
# 
#     return result.tocsr()

