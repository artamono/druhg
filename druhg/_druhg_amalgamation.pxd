# cython: language_level=3
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as np

cdef class Amalgamation (object):
    cdef:
        np.intp_t size
        np.double_t energy
        np.intp_t clusters

        Amalgamation merge_amalgamations(self, np.double_t g, Amalgamation other)
        np.double_t _whole(self, np.double_t g, Amalgamation other)
        np.intp_t limit_to_ought(self, np.double_t g, Amalgamation other)
        void _amalgamate(self, np.intp_t size, np.double_t energy, np.intp_t clusters)
