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

cdef np.double_t EPS = sys.float_info.min

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

    def __init__(self, int size = 1, double energy = 0., int clusters = 1):
        self.size = size
        self.energy = energy
        self.clusters = clusters

    cdef void _amalgamate(self, np.intp_t size, np.double_t energy, np.intp_t clusters):
        # self.energy += energy
        self.energy = merge_means(self.clusters, self.energy, clusters, energy) # храним среднее для лучшей точности
        self.size += size
        self.clusters += clusters

    cdef np.double_t border_overcoming(self, np.double_t g, Amalgamation other):
        # returns negative if clusterization didn't happen
        # limit_to_ought:
        # Die Schranke und das Sollen
        # can a new whole overcome its' parts?
        cdef np.double_t limit, jump

        limit = g * pow(1.*min(self.size, other.size), 0.5)
        # limit *= self.clusters # No need of multiplying when working with merge means
        #* pow((1.*self.clusters + other.clusters)/max(self.clusters, other.clusters), +0.25) - this is an interesting idea, may be it will help us in the future

        jump = -1.
        if limit >= self.energy:
            jump = g * self.size
        # if self.size > 1 or other.size==1:
        #     print (min(self.size, other.size) > 1, other.size, brd.limit > self.energy, self.size, self.clusters, 'dis', brd.dis, 'lim', brd.limit, self.energy )
        return jump

    cdef Amalgamation merge_amalgamations(self, np.double_t g, Amalgamation other, np.double_t jump1, np.double_t jump2):
        cdef:
            np.intp_t osize, oclusters
            np.double_t oenergy
            Amalgamation ret

        ret = self
        if self.size == 1:
            ret = Amalgamation(1, 0., 1)
# ----------------------
        osize, oenergy, oclusters = other.size, other.energy, other.clusters
# ----------------------
        if jump1 >= 0:
            ret.energy = jump1
            ret.clusters = 1
# ----------------------
        if jump2 >= 0:
            oenergy = jump2
            oclusters = 1
# ----------------------

        ret._amalgamate(osize, oenergy, oclusters)
        return ret
