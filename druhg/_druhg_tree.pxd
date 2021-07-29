import numpy as np
cimport numpy as np

cdef struct Relation:
    np.double_t reciprocity
    np.double_t upper_bound
    np.double_t jump
    np.intp_t target
    np.intp_t my_rank
    np.intp_t rec_rank
    np.intp_t my_scope
    np.double_t my_dis
    np.double_t rec_dis
    np.intp_t penalty
    np.intp_t my_members
    np.intp_t rec_members
    np.intp_t loop_rank
    np.intp_t index
    np.double_t value
