# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: initializedcheck=False
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
    meana = meana + delta*nb/(na + nb)
    # use this for big n's
    # mu = (mu*n + mu_2*n_2) / nx
    # m2a = m2a + m2b + delta**2*na*nb/nx
    return meana


cdef class Amalgamation (object):
    # declarations are in pxd file
    # https://cython.readthedocs.io/en/latest/src/userguide/sharing_declarations.html

    def __init__(self, int size = 1, double energy = 0., int clusters = 1):
        self.size = size
        self.energy = energy
        self.clusters = clusters


    cdef np.double_t _whole(self, np.double_t g, Amalgamation other):
        cdef np.double_t quality, quantity, measure

        quality = pow(g, 1.)
        quantity = pow(1.*min(self.size, other.size), 0.5)
        measure = pow((1.*self.clusters + other.clusters)/max(self.clusters, other.clusters), 0.25)

        # print (max(self.size, other.size), min(self.size, other.size), 'whole', 1.*quality*quantity*measure,'=', 1.*self.clusters, quality, quantity, measure)
        return 1.*quality*quantity*measure

    cdef np.intp_t limit_to_ought(self, np.double_t g, Amalgamation other):
        # Die Schranke und das Sollen
        # can a new whole overcome its' parts?
        cdef np.double_t whole

        whole = self._whole(g, other)
        # print (whole > self.energy + EPS, whole, self.energy + EPS)
        return whole > self.energy + EPS

    cdef Amalgamation merge_amalgamations(self, np.double_t g, Amalgamation other):
        cdef np.intp_t osize, oclusters
        cdef np.double_t whole, oenergy
        cdef Amalgamation ret

        ret = self
        if self.size == 1:
            ret = Amalgamation(1, 0., 1)
# ----------------------
        osize, oenergy, oclusters = other.size, other.energy, other.clusters

        whole = self._whole(g, other)
# ----------------------
        if whole > self.energy + EPS:
            ret.energy = whole*ret.size
            ret.clusters = 1
# ----------------------
        if whole > oenergy + EPS:
            oenergy = whole*osize
            oclusters = 1
# ----------------------

        ret._amalgamate(osize, oenergy, oclusters)
        return ret

    cdef void _amalgamate(self, np.intp_t size, np.double_t energy, np.intp_t clusters):
        # self.energy += energy
        self.energy = merge_means(self.clusters, self.energy, clusters, energy)
        self.size += size
        self.clusters += clusters
