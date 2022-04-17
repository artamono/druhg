# cython: language_level=3
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

# amalgamation structure that can become a cluster
# Author: Pavel "DRUHG" Artamonov
# License: 3-clause BSD

import numpy as np
cimport numpy as np
import sys

from collections import Counter
from libc.math cimport fabs, pow

cdef np.double_t merge_means(np.intp_t na, np.double_t meana,
                             np.intp_t nb, np.double_t meanb
                            ):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    # Chan et al.[10] Welford's online algorithm
    cdef np.double_t delta

    # nx = na + nb
    delta = meanb - meana
    delta = meana + delta*nb/(na + nb)
    # use this for big n's
    # mu = (mu*n + mu_2*n_2) / nx
    # m2a = m2a + m2b + delta**2*na*nb/nx
    return delta

cdef class Amalgamation (object):
    # declarations are in pxd file
    # https://cython.readthedocs.io/en/latest/src/userguide/sharing_declarations.html

    def __init__(self, int size = 1, int clusters = 1):
        self.size = size
        self.clusters = clusters
        self.energies = None

    cdef void _amalgamate(self, np.intp_t size, np.intp_t clusters, dict o_energies):
        self.size += size
        self.clusters += clusters

        energy = dict()
        for elem in set(self.energies) | set(o_energies):
            v1 = self.energies.get(elem, (0,0.,))
            v2 = o_energies.get(elem, (0,0.,))
            energy[elem] = (v1[0]+v2[0], v1[1]+v2[1],)

        self.energies = energy

    cdef Amalgamation merge_amalgamations(self, np.double_t g, Amalgamation other, np.double_t jump1, np.double_t jump2):
        cdef:
            np.intp_t osize, oclusters
            Amalgamation ret
        ret = self
        if self.size == 1:
            ret = Amalgamation() # the self will be reused
# ----------------------
        osize, oclusters, o_energies = other.size, other.clusters, other.energies
# ----------------------
        if jump1 >= 0:
            ret.clusters = 1
            ret.energies = {ret.size : (1., jump1,)}
# ----------------------
        if jump2 >= 0:
            oclusters = 1
            o_energies = {osize : (1., jump2,)}
# ----------------------

        ret._amalgamate(osize, oclusters, o_energies)
        return ret


    cdef np.double_t border_overcoming(self, np.double_t g, Amalgamation other, np.double_t PRECISION):
        # returns negative if cluster didn't form
        # limit_to_ought:
        # Die Schranke und das Sollen
        # can a new whole overcome its' parts?
        cdef np.double_t insides, limit
        cdef np.double_t clusters, osize
        cdef np.double_t n, c, e

        if self.energies is None:
            return g # first connection always clusterize

        clusters = self.clusters - 1
        osize = other.size

        insides, limit = 0., 0.
        for n in self.energies:
            c,e = self.energies[n]
            insides += n/min(n, clusters) * e
            limit += min(n, osize)/n * c
            # print (n, e, c, n/min(n, clusters) * e, min(n, osize)/n * c)

        limit *= g-PRECISION

        # print('..',g, 'c', clusters, other.clusters, 's', self.size, other.size, 'r', insides,  limit)
        # print(list(self.energies))
        # print (insides > limit,'=============================================')

        if insides < limit:
            return g

        return -1.

    cdef np.double_t border_overcoming_rev(self, np.double_t g, Amalgamation other, np.double_t PRECISION):
        # returns negative if cluster didn't form
        # limit_to_ought:
        # Die Schranke und das Sollen
        # can a new whole overcome its' parts?
        cdef np.double_t insides, limit
        cdef np.double_t clusters, osize
        cdef np.double_t n, c, e

        if self.energies is None:
            return pow(g, -1) # first connection always clusterize

        clusters = self.clusters - 1
        osize = other.size

        insides, limit = 0., 0.
        for n in self.energies:
            c,e = self.energies[n]
            insides += min(n, clusters)/n * e
            limit += n/min(n, osize) * c

        insides *= g-PRECISION
        if insides > limit:
            return pow(g, -1)

        return -1.
