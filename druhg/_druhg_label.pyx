# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# Labels nodes given mst-edgepairs and cluster-parents of DRUHG's results.
# Also provides tools for label manipulations, such as:
# * Treats small clusters as outliers on-demand
# * Breaks big clusters on-demand
# * Glues outliers to the nearest clusters on-demand
#
# Author: Pavel "DRUHG" Artamonov
# License: 3-clause BSD


import numpy as np
cimport numpy as np

cdef class UnionFind (object):

    cdef np.ndarray parent_arr
    cdef np.ndarray size_arr

    cdef np.intp_t next_label
    cdef np.intp_t *parent
    cdef np.intp_t *size

    cdef np.intp_t full_size
    cdef np.intp_t upper_size_limit

    cdef set cluster_parents

    def __init__(self, N, parents, edges_arr, num_edges, upper_size_limit):
        self.full_size = N
        self.next_label = N + 1

        self.upper_size_limit = upper_size_limit

        self.parent_arr = np.zeros(2 * N, dtype=np.intp)
        self.size_arr = np.ones(N, dtype=np.intp)

        self.parent = (<np.intp_t *> self.parent_arr.data)
        self.size = (<np.intp_t *> self.size_arr.data)

        self.cluster_parents = set(parents)

        self._fill_structure(edges_arr, num_edges)

    cdef void _fill_structure(self, np.ndarray edges, np.intp_t num_edges):
        cdef np.intp_t i

        for i in range(0, num_edges):
            self.merge(edges[2*i], edges[2*i + 1])

    cdef np.intp_t fast_find_parents(self, np.intp_t n):
        cdef np.intp_t p, temp, lab

        p = self.parent[n]
        if p == 0:
            return n
        lab = 0
        if p in self.cluster_parents and self.size[p - self.full_size] < self.upper_size_limit:
            lab = p
        while self.parent[p] != 0:
            p = self.parent[p]
            if p in self.cluster_parents and self.size[p - self.full_size] < self.upper_size_limit:
                lab = p
        if lab != 0:
            while lab != n:
                temp = self.parent[n]
                self.parent[n] = lab
                n = temp
        return p

    cdef void union(self, np.intp_t aa, np.intp_t bb):
        self.parent[aa] = self.parent[bb] = self.next_label
        self.next_label += 1
        return

    cdef void merge(self, np.intp_t a, np.intp_t b):
        cdef np.intp_t aa, bb, i

        i = self.next_label - self.full_size
        aa, bb = self.fast_find_parents(a), self.fast_find_parents(b)

        a = (self.parent[a] != 0)*(aa - self.full_size)
        b = (self.parent[b] != 0)*(bb - self.full_size)
        self.union(aa, bb)

        self.size[i] = self.size[a] + self.size[b]
        return

cdef void fixem(UnionFind U, np.ndarray edges_arr, np.intp_t num_edges, np.ndarray result):
    cdef np.intp_t p, a, b

    new_results = set()
    new_path = []
    restart = []
    for p in range(0, num_edges):
        a, b = edges_arr[2*p], edges_arr[2*p + 1]
        if result[a] < 0 and result[b] < 0:
            new_results.update([a,b])
            new_path.append((a,b))
            continue
        elif result[b] < 0:
            a,b = b,a
        elif result[a] >= 0:
            continue
        res = result[b]
        result[a] = res
        if a in new_results:
            links = set([a])
            dontstop = 1
            while dontstop:
                dontstop = 0
                for path in list(new_path):
                    a, b = path
                    if a in links or b in links:
                        result[a] = result[b] = res
                        # print ('new_r', a, b, res)
                        links.update([a,b])
                        new_path.remove(path)
                        dontstop = 1

    return

cdef void fixem_old(np.ndarray edges_arr, np.intp_t num_edges, np.ndarray result):
    cdef np.intp_t p, a, b
    restart = []
    for p in range(0, num_edges):
        a, b = edges_arr[2*p], edges_arr[2*p + 1]
        if result[a] < 0 and result[b] < 0:
            restart.append(p)
        elif result[a] < 0:
            result[a] = result[b]
        elif result[b] < 0:
            result[b] = result[a]
    restart2 = []
    while len(restart)!=len(restart2):
        restart2 = restart
        restart = []
        for p in restart2:
            a, b = edges_arr[2*p], edges_arr[2*p + 1]
            if result[a] < 0 and result[b] < 0:
                restart.append(p)
            elif result[a] < 0:
                result[a] = result[b]
            elif result[b] < 0:
                result[b] = result[a]
    return


cpdef np.ndarray do(np.intp_t size, np.ndarray edges_arr, np.intp_t num_edges, np.ndarray parents_arr, np.intp_t limit1 = 0, np.intp_t limit2 = 0, np.intp_t fix_outliers = 0):
    """Returns cluster labels.
    
    Uses the results of DRUHG algorithm(edges and parents).
    Marks data-points with corresponding parent index of a cluster.
    The parameters `limit1` and 'limit2' allows the clustering to declare noise points.
    Outliers-noise marked by -1.

    Parameters
    ----------
    size : int
        Amount of nodes.
    
    edges_arr : ndarray
        Edgepairs of mst.
    
    num_edges : int
        Amount of edges.
    
    parents_arr : ndarray
        Parent-indexes of unionfind. Could be used to surgically removing clusters.  
    
    limit1 : int, optional (default=sqrt(size))
        Clusters that are smaller than this limit treated as noise. 
        Use 1 to find True outliers.
        
    limit2 : int, optional (default=size)
        Clusters with size OVER this limit treated as noise. 
        Use it to break down big clusters.
 
    fix_outliers: int, optional (default=0)
        All outliers will be assigned to the nearest cluster 

    Returns
    -------

    labels : array [size]
       An array of cluster labels, one per data-point. Unclustered points get
       the label -1.
    """

    cdef np.intp_t i, p, s
    cdef np.ndarray[np.intp_t, ndim=1] result_arr

    if limit1 <= 0:
        limit1 = int(np.ceil(np.sqrt(size)))
    if limit2 <= 0:
        limit2 = size

    result_arr = -1*np.ones(size, dtype=np.intp)
    result = (<np.intp_t *> result_arr.data)

    U = UnionFind(size, parents_arr, edges_arr, num_edges, limit2)

    i = size
    while i:
        i -= 1
        U.fast_find_parents(i)

        p = U.parent[i]
        s = (p in U.cluster_parents)*U.size[p - size] + 1*(p not in U.cluster_parents)
        if limit1 < s < limit2:
            result[i] = p

    if fix_outliers != 0 and len(np.unique(result_arr))>1:
        fixem(U, edges_arr, num_edges, result_arr)

    return result_arr


cdef np.ndarray pretty(np.ndarray labels_arr):
    """ Relabels to pretty positive integers. 
    """
    cdef np.intp_t i, p, label, max_label

    cdef np.ndarray[np.intp_t, ndim=1] result_arr

    result_arr = -1*np.ones(len(labels_arr), dtype=np.intp)
    result = (<np.intp_t *> result_arr.data)

    converter = {-1: -1}
    max_label = 0
    i = len(labels_arr)
    while i:
        i -= 1
        p = labels_arr[i]
        if p in converter:
            label = converter[p]
        else:
            label = max_label
            converter[p] = max_label
            max_label += 1
        result[i] = label

    return result_arr
