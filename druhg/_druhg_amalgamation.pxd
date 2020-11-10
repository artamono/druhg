import numpy as np
cimport numpy as np


cdef class Amalgamation (object):
    cdef np.intp_t size
    cdef np.double_t energy
    cdef np.intp_t clusters

    cdef Amalgamation merge_amalgamations(self, np.double_t g, Amalgamation other)
    cdef np.double_t _whole(self, np.double_t g, Amalgamation other)
    cdef np.intp_t limit_to_ought(self, np.double_t g, Amalgamation other)
    cdef void _amalgamate(self, np.intp_t size, np.double_t energy, np.intp_t clusters)
