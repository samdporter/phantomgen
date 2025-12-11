import numpy as np
import pytest

from phantomgen import create_nema
from phantomgen.core import _downsample_volume
from phantomgen.presets import pet_nema_dict


def _make_phantom(matrix_size, voxel_size_mm, supersample):
    return create_nema(
        matrix_size=matrix_size,
        voxel_size_mm=voxel_size_mm,
        nema_dict=pet_nema_dict,
        supersample=supersample,
    )


def _activity_totals(matrix_size, voxel_size_mm, factors):
    totals = []
    for factor in factors:
        activity, _ = _make_phantom(matrix_size, voxel_size_mm, supersample=factor)
        totals.append(activity.sum(dtype=np.float64))
    return totals


def _attenuation_means(matrix_size, voxel_size_mm, factors):
    means = []
    for factor in factors:
        _, attenuation = _make_phantom(matrix_size, voxel_size_mm, supersample=factor)
        means.append(attenuation.mean(dtype=np.float64))
    return means


def test_total_activity_highres_conserved():
    matrix_size = (128, 160, 128)
    voxel_size_mm = (2.0, 2.0, 2.0)
    totals = _activity_totals(matrix_size, voxel_size_mm, factors=[1, 2])
    reference = totals[0]
    for total in totals[1:]:
        assert np.isclose(total, reference, rtol=1.5e-2, atol=10.0)


def test_total_activity_coarse_vs_supersampled():
    matrix_size = (72, 80, 72)
    voxel_size_mm = (3.5, 4.0, 3.5)
    base, refined = _activity_totals(matrix_size, voxel_size_mm, factors=[1, 3])
    # Expect close agreement (activity should be conserved) but allow a modest tolerance
    assert np.isclose(base, refined, rtol=3e-2, atol=1.0)


def test_attenuation_mean_highres_conserved():
    matrix_size = (128, 160, 128)
    voxel_size_mm = (2.0, 2.0, 2.0)
    means = _attenuation_means(matrix_size, voxel_size_mm, factors=[1, 2])
    reference = means[0]
    for mu in means[1:]:
        assert np.isclose(mu, reference, rtol=1e-2, atol=1e-3)


def test_attenuation_mean_coarse_vs_supersampled():
    matrix_size = (72, 80, 72)
    voxel_size_mm = (3.5, 4.0, 3.5)
    coarse, refined = _attenuation_means(matrix_size, voxel_size_mm, factors=[1, 3])
    assert np.isclose(coarse, refined, rtol=5e-3, atol=5e-4)


@pytest.mark.parametrize(
    "supersample",
    [
        (2, 2, 2),
        (3, 1, 1),
        (1, 2, 3),
    ],
)
def test_supersample_matches_manual_downsample(supersample):
    base_matrix = (72, 80, 72)
    base_voxel = (3.5, 4.0, 3.5)

    act_super, ct_super = _make_phantom(
        base_matrix, base_voxel, supersample=supersample
    )

    high_matrix = tuple(m * f for m, f in zip(base_matrix, supersample))
    high_voxel = tuple(v / f for v, f in zip(base_voxel, supersample))

    act_high, ct_high = _make_phantom(high_matrix, high_voxel, supersample=1)

    act_down = _downsample_volume(act_high, supersample, reduce="sum")
    ct_down = _downsample_volume(ct_high, supersample, reduce="mean")

    assert np.allclose(act_super, act_down, rtol=1e-5, atol=1e-5)
    assert np.allclose(ct_super, ct_down, rtol=1e-5, atol=1e-5)
