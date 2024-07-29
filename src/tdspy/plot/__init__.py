try:
    import matplotlib
except ImportError as e:
    print("Please install suitable version of matplotlib.")
    raise e

from .eigenvalues import eigen_plot
from .discretization_animation import discretization_animation