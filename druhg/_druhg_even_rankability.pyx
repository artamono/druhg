# cython: boundscheck=False
# cython: nonecheck=False
# cython: initializedcheck=False
# DRUHG
# even subjective rankability distance computations
# Authors: Pavel "DRUHG" Artamonov
# License: 3-clause BSD

import numpy as np
cimport numpy as np

import sys

from scipy.sparse import lil_matrix as sparse_matrix
import gc

def even_rankability(distance_matrix, min_flatting=0, max_neighbors_search=0, step_ranking=0, step_factor=0,
                     distance_factor=0.0, is_ranks=False, verbose=True):
    """Compute the weighted adjacency matrix of the even rankability
    graph of a distance matrix.

    Parameters
    ----------
    distance_matrix : ndarray, shape (n_samples, n_samples)
        Array of distances between samples.

    step_ranking : nint, optional (default=0)
        The step_ranking parameter of DRUHG - how many ranks to downgrade the even subjective ranking

    min_flatting : nint, optional (default=0)
        The min_flatting paramater of DRUHG - how many neighbors of the point to flat rank and even their distances

    max_neighbors_search: nint, optional (default=0)
        The max_neighbors_search parameter of DRUHG - how many neighbors of the point to match their ranks with

    is_ranks: boolean, optional (default=0)
        Ranks instead of distances

    step_factor: nint, optional (default=0)
        Rank downgrade factor coefficient

    distance_factor: float, optional (default=0.)
        Distance factor to extent neighbors ranks

    Returns
    -------
    even_rankability: ndarray, shape (n_samples, n_samples)
        Weighted adjacency matrix of the even rankability graph.

    """

    assert (distance_matrix.shape[0] == distance_matrix.shape[1])

    size = distance_matrix.shape[0]
    if max_neighbors_search <= 0:
        max_neighbors_search = size - 1
    max_neighbors_search = np.min([max_neighbors_search, size - 1])

    if min_flatting > size - 1:
        min_flatting = size - 1  # this will give a funny result )))

    result = np.copy(distance_matrix)

    # sorting is not working properly because of double to float cython convertion, argsort is ok
    sortedM = np.argsort(result, axis=1)

    if is_ranks:
        result.fill(size)

    progress = -1
    if verbose:
        sys.stderr.write('progress: ', )

    for i in range(1, size):
        if verbose:
            newV = 100 * i // size // 5
            if progress < newV:
                progress = newV
                if progress % 2:
                    sys.stderr.write('%', )
                else:
                    sys.stderr.write(str(progress * 5), )

        for ki in range(1, max_neighbors_search + 1):
            j = sortedM[i, ki]

            distance = distance_matrix[i][j]

            rank_i = 0
            rank_j = 0

            if distance_factor == 0.0:
                rank_i = ki
                rank_j = np.where(sortedM[j] == i)[0][0]

                while distance_matrix[i][sortedM[i][rank_i - 1]] == distance:
                    rank_i -= 1
                while distance_matrix[j][sortedM[j][rank_j - 1]] == distance:
                    rank_j -= 1
            else:
                distance = distance + distance * distance_factor
                rank_i = size - 1
                for ii in range(ki, size):
                    if distance_matrix[i][sortedM[i][ii]] > distance:
                        rank_i = ii - 1
                        break
                rank_j = size - 1
                for ii in range(ki, size):
                    if distance_matrix[j][sortedM[j][ii]] > distance:
                        rank_j = ii - 1
                        break

            rank_max = rank_j

            if rank_i > rank_j:
                rank_max = rank_i

            rank_max += step_ranking + step_factor * rank_max
            if rank_max >= size:
                rank_max = size - 1

            if rank_max < min_flatting:
                rank_max = min_flatting

            if is_ranks:
                result[i][j] = result[j][i] = rank_max
            else:
                result[i][j] = result[j][i] = np.max(
                    [distance_matrix[j][sortedM[j][rank_max]], distance_matrix[i][sortedM[i][rank_max]]])

    if verbose:
        sys.stderr.write(str(100) + '% ', )
    return result  #net_ranks#, pushed_ranks, pusher_ranks,

def even_rankability_old(distance_matrix, min_flatting=0, verbose = True):
    # no max_neighbors_search
    """Compute the weighted adjacency matrix of the even rankability
    graph of a distance matrix.

    Parameters
    ----------
    distance_matrix : ndarray, shape (n_samples, n_samples)
        Array of distances between samples.

    min_flatting : nint, optional (default=0)
        The min_flatting paramater of DRUHG - how many neighbors of the point to flat rank and even their distances

    max_neighbors_search: nint, optional (default=0)
        The max_neighbors_search parameter of DRUHG - how many neighbors of the point to match their ranks with


    Returns
    -------
    even_rankability: ndarray, shape (n_samples, n_samples)
        Weighted adjacency matrix of the even rankability graph.

    pushed_ranks: ndarray, shape (n_samples)
        Vector of cummulated pushed ranks

    ranks_equalized: int
        Total amount of ranks changed during comparison

    """

    assert (distance_matrix.shape[0] == distance_matrix.shape[1])

    size = distance_matrix.shape[0]

    if min_flatting > size - 1:
        min_flatting = size - 1  # this will give a funny result )))

    result = np.copy(distance_matrix)
    net_ranks = 1

    # sorting is not working properly because of double to float cython convertion, argsort is ok
    sortedM = np.argsort(result, axis=1)

    progress = -1

    for i in range(size - 1):
        if verbose:
            newV = 100 * i // size // 10
            if progress < newV:
                progress = newV
                sys.stderr.write(str(progress * 10) + '%', )

        for j in range(i + 1, size):
            distance = distance_matrix[i][j]

            # rank_i = len(distance_matrix[distance_matrix[i]<distance])
            # rank_j = len(distance_matrix[distance_matrix[j]<distance])

            rank_i = np.where(sortedM[i] == j)[0][0]
            rank_j = np.where(sortedM[j] == i)[0][0]

            while (distance_matrix[i][sortedM[i][rank_i - 1]] == distance):
                rank_i -= 1
            while (distance_matrix[j][sortedM[j][rank_j - 1]] == distance):
                rank_j -= 1

            if rank_i > rank_j:
                rank = rank_i

                if rank < min_flatting:
                    rank = min_flatting
                if distance != distance_matrix[j][sortedM[j][rank]]:
                    result[i][j] = result[j][i] = distance_matrix[j][sortedM[j][rank]]
                    # rr = 0
                    # rank_j += 1
                    # while (rank_j <= rank):
                    #     if (distance != distance_matrix[j][sortedM[j][rank_j]]):
                    #         rr += 1
                    #     rank_j += 1
                    # # pushed_ranks[i] += rr
                    # # pusher_ranks[j] += rr
                    # net_ranks += rr
            else:
                rank = rank_j

                if rank < min_flatting:
                    rank = min_flatting
                if distance != distance_matrix[i][sortedM[i][rank]]:
                    result[i][j] = result[j][i] = distance_matrix[i][sortedM[i][rank]]
                    # rr = 0
                    # rank_i += 1
                    # while (rank_i <= rank):
                    #     if (distance != distance_matrix[i][sortedM[i][rank_i]]):
                    #         rr += 1
                    #     rank_i += 1
                    # # pushed_ranks[j] += rr
                    # # pusher_ranks[i] += rr
                    # net_ranks += rr

            # distance = np.max([distance_matrix[i][sortedM[i][rank]],distance_matrix[j][sortedM[j][rank]]])
    if verbose:
        sys.stderr.write(str(100) + '% ', )
    return result, net_ranks  #, pushed_ranks, pusher_ranks,

def even_rankability_ranks(distance_matrix, min_flatting=0):
    """Compute the weighted adjacency matrix of the even rankability
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
        Weighted adjacency matrix of the even rankability graph.

    pushed_ranks: ndarray, shape (n_samples)
        Vector of cummulated pushed ranks

    ranks_equalized: int
        Total amount of ranks changed during comparison

    """

    assert (distance_matrix.shape[0] == distance_matrix.shape[1])

    size = distance_matrix.shape[0]

    if min_flatting > size:
        min_flatting = size  # this will give a funny result )))

    result = np.copy(distance_matrix)
    net_ranks = 0

    # sorting is not working properly because of double to float cython convertion, argsort is ok
    sortedM = np.argsort(result, axis=1)

    for i in range(0, size - 1):
        for j in range(i + 1, size):
            distance = distance_matrix[i][j]

            # rank_i = len(distance_matrix[distance_matrix[i]<distance])
            # rank_j = len(distance_matrix[distance_matrix[j]<distance])

            rank = 0
            rr = 0

            rank_i = np.where(sortedM[i] == j)[0][0]
            rank_j = np.where(sortedM[j] == i)[0][0]

            while (distance_matrix[i][sortedM[i][rank_i - 1]] == distance):
                rank_i -= 1
            while (distance_matrix[j][sortedM[j][rank_j - 1]] == distance):
                rank_j -= 1

            # if rank_i > rank_j:
            #     rank = rank_i
            #
            #     if rank < min_flatting:
            #         rank = min_flatting
            # if (distance != distance_matrix[j][sortedM[j][rank]]):
            #     result[i][j] = result[j][i] = distance_matrix[j][sortedM[j][rank]]
            # rank_j += 1
            # while (rank_j <= rank):
            #     if (distance != distance_matrix[j][sortedM[j][rank_j]]):
            #         rr += 1
            #     rank_j += 1
            # pushed_ranks[i] += rr
            # pusher_ranks[j] += rr
            # else:
            #     rank = rank_j
            #
            #     if rank < min_flatting:
            #         rank = min_flatting
            # if (distance != distance_matrix[i][sortedM[i][rank]]):
            #     result[i][j] = result[j][i] = distance_matrix[i][sortedM[i][rank]]
            # rr = 0
            # rank_i += 1
            # while (rank_i <= rank):
            #     if (distance != distance_matrix[i][sortedM[i][rank_i]]):
            #         rr += 1
            #     rank_i += 1
            # pushed_ranks[j] += rr
            # pusher_ranks[i] += rr

            # net_ranks += rr
            result[i][j] = result[j][i] = np.max(rank_i, rank_j, min_flatting)
            # distance = np.max([distance_matrix[i][sortedM[i][rank]],distance_matrix[j][sortedM[j][rank]]])

    return result, net_ranks  #, pushed_ranks, pusher_ranks,

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

cpdef np.ndarray[np.double_t, ndim=2] mst_linkage_core(
        np.ndarray[np.double_t,
                   ndim=2] distance_matrix):
    cdef np.ndarray[np.intp_t, ndim=1] node_labels
    cdef np.ndarray[np.intp_t, ndim=1] current_labels
    cdef np.ndarray[np.double_t, ndim=1] current_distances
    cdef np.ndarray[np.double_t, ndim=1] left
    cdef np.ndarray[np.double_t, ndim=1] right
    cdef np.ndarray[np.double_t, ndim=2] result

    cdef np.ndarray label_filter

    cdef np.intp_t current_node
    cdef np.intp_t new_node_index
    cdef np.intp_t new_node
    cdef np.intp_t i
    cdef np.intp_t num_edges

    result = np.zeros((distance_matrix.shape[0] - 1, 3))
    node_labels = np.arange(distance_matrix.shape[0], dtype=np.intp)
    current_node = 0
    current_distances = np.infty * np.ones(distance_matrix.shape[0])
    current_labels = node_labels
    num_edges = node_labels.shape[0] - 1

    for i in range(num_edges):
        label_filter = current_labels != current_node
        current_labels = current_labels[label_filter]
        left = current_distances[label_filter]
        right = distance_matrix[current_node][current_labels]
        current_distances = np.where(left < right, left, right)

        new_node_index = np.argmin(current_distances)
        new_node = current_labels[new_node_index]
        result[i, 0] = <double> current_node
        result[i, 1] = <double> new_node
        result[i, 2] = current_distances[new_node_index]
        current_node = new_node

    return result
