# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Adam Peichl
# Copyright (C) 2026 Adrian Saldanha

try:
    import matplotlib
except ImportError as e:
    print("Please install suitable version of matplotlib.")
    raise e

from .eigenvalues import eigen_plot, complex_scatter_axplot
from .discretization_animation import discretization_animation