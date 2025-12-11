import numpy as np
from phantomgen import create_nema, pet_nema_dict

def test_shapes_and_values():
    act, ct = create_nema(matrix_size=(64,64,64), voxel_size_mm=(8,8,8), nema_dict=pet_nema_dict)
    assert act.shape == ct.shape == (64,64,64)
    assert act.dtype == np.float32 and ct.dtype == np.float32
    # Should have nonzero fill in expected ranges
    assert np.isfinite(act).all() and act.max() > 0
    assert np.isfinite(ct).all() and ct.max() > 0


def test_create_nema_with_none_dict():
    """Test that create_nema works with nema_dict=None (uses defaults)."""
    act, ct = create_nema(matrix_size=(64, 64, 64), voxel_size_mm=(8, 8, 8), nema_dict=None)
    assert act.shape == ct.shape == (64, 64, 64)
    assert act.dtype == np.float32 and ct.dtype == np.float32
    assert np.isfinite(act).all() and act.max() > 0
    assert np.isfinite(ct).all() and ct.max() > 0
