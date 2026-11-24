"""Shared fixtures. All test data are SYNTHETIC."""
import pytest

from read_eareog.synthetic import synthetic_cohort


@pytest.fixture(scope='session')
def cohort():
    return synthetic_cohort(4, seed=21, minutes=2.0)
