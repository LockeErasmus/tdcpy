"""
Pytest configuration
"""

import os
import sys

import pytest
import numpy as np

repository_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
source_path = os.path.join(repository_path, "src")
#sys.path.append(repository_path)
sys.path.append(source_path)


# setup for nicer outputs
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
