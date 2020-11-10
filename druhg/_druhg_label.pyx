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

from _druhg_amalgamation cimport Amalgamation
from _druhg_amalgamation import Amalgamation

cdef class UnionFind (object):

    cdef np.ndarray parent_arr
    cdef np.intp_t *parent

    cdef np.intp_t next_label
    cdef np.intp_t full_size

    def __init__(self, N):
        self.full_size = N
        self.next_label = N + 1

        self.parent_arr = np.zeros(2 * N, dtype=np.intp)
        self.parent = (<np.intp_t *> self.parent_arr.data)

        # self._fill_structure(edges_arr, num_edges)

    cdef np.intp_t has_parent(self, np.intp_t n):
        return self.parent[n]

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

    cdef np.intp_t passive_find(self, np.intp_t n):
        cdef np.intp_t p, temp

        p = self.parent[n]
        if p == 0:
            return n
        while self.parent[p] != 0:
            p = self.parent[p]

        return p

    cdef np.intp_t passive_union(self, np.intp_t aa, np.intp_t bb):
        # aa, bb = self.fast_find(aa), self.fast_find(bb)

        ret = self.next_label
        self.parent[aa] = self.parent[bb] = ret
        self.next_label += 1
        return ret

    cdef list discard_clusters(self, np.intp_t n, set clusters):
        cdef np.intp_t p, temp
        cdef list ret

        ret = []
        p = self.parent[n]
        if p == 0:
            return ret
        while self.parent[p] != 0:
            if p in clusters:
                ret.append(p)
            p = self.parent[p]
        if ret:
            ret.pop()
        return ret

    cdef np.intp_t label_find(self, np.intp_t n, set clusters):
        cdef np.intp_t p, temp, label

        label = -1

        p = self.parent[n]
        if p == 0:
            return label

        if p in clusters:
            label = p
        while self.parent[p] != 0:
            p = self.parent[p]
            if p in clusters:
                label = p
        if label >= 0:
            while label != n:
                temp = self.parent[n]
                self.parent[n] = label
                n = temp
        return label

cdef void fixem(np.ndarray edges_arr, np.intp_t num_edges, np.ndarray result):
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

cdef set emerge_clusters(UnionFind U, np.ndarray edges_arr, np.ndarray values_arr, np.intp_t limit1, np.intp_t limit2, list exclude):

    cdef np.intp_t e1,e2, p1,p2, i, c
    cdef np.double_t v
    cdef Amalgamation being, being1, being2, being3
    cdef list disc
    cdef dict d
    cdef set clusters

    being = Amalgamation()
    being1 = being
    being2 = being

    d = {}
    clusters = set()

    for i, v in enumerate(values_arr):
        e1, e2 = edges_arr[2*i], edges_arr[2*i+1]
        # print ('edges', e1, e2)
        p1, p2 = U.passive_find(e1), U.passive_find(e2)
        being1, being2 = being, being

        if U.has_parent(e1):
            being1 = d.pop(p1)
        if U.has_parent(e2):
            being2 = d.pop(p2)

        if being1.size > 1 \
            and being1.limit_to_ought(v, being2) \
            and limit1 < being1.size < limit2 \
            and p1 not in exclude:
            clusters.add(p1)

        if being2.size > 1 \
            and being2.limit_to_ought(v, being1) \
            and limit1 < being2.size < limit2 \
            and p2 not in exclude:
            clusters.add(p2)

        being3 = being1.merge_amalgamations(v, being2)
        e3 = U.passive_union(U.passive_find(e1), U.passive_find(e2))
        # print(p1,p2,e3)
        d[e3] = being3
        if being2 != being:
            del being2

        disc = U.discard_clusters(e1, clusters)
        for c in disc:
            clusters.discard(c)
        disc = U.discard_clusters(e2, clusters)
        for c in disc:
            clusters.discard(c)

        U.label_find(e1, clusters)
        U.label_find(e2, clusters)

    return clusters


cpdef np.ndarray label(np.ndarray edges_arr, np.ndarray values_arr, int size = 0, list exclude = None, np.intp_t limit1 = 0, np.intp_t limit2 = 0, np.intp_t fix_outliers = 1):
    """Returns cluster labels.
    
    Uses the results of DRUHG MST-tree algorithm(edges and values).
    Marks data-points with corresponding parent index of a cluster.
    Exclude list breaks passed clusters by their parent index.
    The parameters `limit1` and 'limit2' allows the clustering to declare noise points.
    Outliers-noise marked by -1.

    Parameters
    ----------
    edges_arr : ndarray
        Edgepair nodes of mst.
    
    values_arr : ndarray
        Edge values.
        
    size : int
        Amount of nodes.
        
    exclude : list
        Clusters with parent-index from this list will not be formed. 
    
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

    cdef set clusters
    cdef int i

    cdef np.ndarray result_arr
    cdef np.intp_t *result


    if limit1 <= 0:
        limit1 = int(np.ceil(np.sqrt(size)))
        print ('label: default value for limit1 is used, clusters below '+str(limit1)+' are considered as noise.')

    if limit2 <= 0:
        limit2 = size
        print ('label: default value for limit2 is used, clusters above '+str(limit2)+' will not be formed.')

    if size == 0:
        size = int(len(edges_arr)/2 + 1)

    if not exclude:
        exclude = []

    result_arr = -1*np.ones(size, dtype=np.intp)
    result = (<np.intp_t *> result_arr.data)

    U = UnionFind(size)

    clusters = emerge_clusters(U, edges_arr, values_arr, limit1, limit2, exclude)
    
    i = size
    while i:
        i -= 1
        result[i] = U.label_find(i, clusters)

    if fix_outliers != 0 and len(np.unique(result_arr))>1:
        fixem(edges_arr, len(values_arr), result_arr)

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
