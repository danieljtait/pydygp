"""
Daniel Tait
"""

from .quadop import QuadOperator

from .ns_util import (Interval,
                      get_times,
                      f_quad_xop_interval,
                      b_quad_xop_interval,
                      f_quad_gop_interval,
                      quad_xop_intervals,
                      quad_gop_intervals)


from .linalg_util import (block_diag_intersect_f,
                          vecx_to_ravx,
                          commutation_matrix)
