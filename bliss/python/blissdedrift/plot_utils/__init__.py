
# from .plot_hits import plot_hits
from ._encodings import get_hits_list

from .plot_hits import plot_hits

try:
    from .scatter_matrix import scatter_matrix_hits
except ImportError:
    print("WARN: plot_utils could not import `altair`. `scatter_matrix_hits` will not be available")
