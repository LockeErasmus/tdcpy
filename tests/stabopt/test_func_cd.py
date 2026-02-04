"""
Docstring for test.stability.test_gamma_r
"""

import logging
import pytest

from tests.examples import TDS_CONTROL_MANUAL_NEUTRAL_CASES

@pytest.mark.parametrize(
    argnames="factory",
    argvalues=TDS_CONTROL_MANUAL_NEUTRAL_CASES,
)
def test_func_cd(factory, enable_plot: bool) -> None:
    from tdspy.common.delay_difference_equation import ndde_to_diff, normalize_diff
    from tdspy.stability.gamma_r import gamma_diff, gamma_normalized_diff
    from tdspy.spectral_abscissa import spectral_abscissa_diff

    A, hA, H, hH = factory

    D, hD = ndde_to_diff(H, hH)
    # normalization is not necessary for NDDE
    print(D, hD)

    val, info = spectral_abscissa_diff(H, hH)
    
    print(val, info)

    if enable_plot:
        raise NotImplementedError("Plotting not implemented yet for gamma_r tests.")