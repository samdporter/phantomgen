"""Test that ring_x, ring_y, and ring_z positioning work correctly."""

import numpy as np
import pytest
from phantomgen import create_nema


def get_sphere_center_mm(volume, voxel_size_mm, matrix_size):
    """Helper to find sphere center in mm coordinates."""
    sphere_indices = np.where(volume > 0)
    if len(sphere_indices[0]) == 0:
        return None
    
    z_idx = np.mean(sphere_indices[0])
    y_idx = np.mean(sphere_indices[1])
    x_idx = np.mean(sphere_indices[2])
    
    # Convert indices to mm (centered at origin)
    z_mm = (z_idx - (matrix_size[0] - 1) / 2.0) * voxel_size_mm[0]
    y_mm = (y_idx - (matrix_size[1] - 1) / 2.0) * voxel_size_mm[1]
    x_mm = (x_idx - (matrix_size[2] - 1) / 2.0) * voxel_size_mm[2]
    
    return z_mm, y_mm, x_mm


def test_ring_z_positioning():
    """Test that ring_z controls the z-position of spheres."""
    base_config = {
        'mu_values': {
            'perspex_mu_value': 0.1,
            'fill_mu_value': 0.096,
            'lung_mu_value': 0.029
        },
        'activity_concentration_background': 0,
        'include_lung_insert': False,
        'sphere_dict': {
            'ring_R': 57,
            'ring_z': 0,
            'ring_x': 0,
            'ring_y': 0,
            'spheres': {
                'diametre_mm': [10],
                'angle_loc': [0],
                'act_conc_MBq_ml': [0.4]
            }
        }
    }
    
    matrix_size = (128, 128, 128)
    voxel_size_mm = (2, 2, 2)
    
    # Test default z=0
    act1, _ = create_nema(matrix_size=matrix_size, voxel_size_mm=voxel_size_mm, nema_dict=base_config)
    z1, y1, x1 = get_sphere_center_mm(act1, voxel_size_mm, matrix_size)
    
    # Test z=-40
    config_z = base_config.copy()
    config_z['sphere_dict'] = base_config['sphere_dict'].copy()
    config_z['sphere_dict']['ring_z'] = -40
    
    act2, _ = create_nema(matrix_size=matrix_size, voxel_size_mm=voxel_size_mm, nema_dict=config_z)
    z2, y2, x2 = get_sphere_center_mm(act2, voxel_size_mm, matrix_size)
    
    # Z should change by 40mm, X and Y should stay the same
    assert abs(z2 - z1 - (-40)) < 3, f"ring_z offset failed: expected -40mm, got {z2-z1:.1f}mm"
    assert abs(y2 - y1) < 3, f"Y position changed when it shouldn't: {y1:.1f} -> {y2:.1f}"
    assert abs(x2 - x1) < 3, f"X position changed when it shouldn't: {x1:.1f} -> {x2:.1f}"


def test_ring_y_positioning():
    """Test that ring_y controls the y-position of the ring center."""
    base_config = {
        'mu_values': {
            'perspex_mu_value': 0.1,
            'fill_mu_value': 0.096,
            'lung_mu_value': 0.029
        },
        'activity_concentration_background': 0,
        'include_lung_insert': False,
        'sphere_dict': {
            'ring_R': 57,
            'ring_z': 0,
            'ring_x': 0,
            'ring_y': 0,
            'spheres': {
                'diametre_mm': [10],
                'angle_loc': [0],  # 0 degrees = +X direction
                'act_conc_MBq_ml': [0.4]
            }
        }
    }
    
    matrix_size = (128, 128, 128)
    voxel_size_mm = (2, 2, 2)
    
    # Test default y=0
    act1, _ = create_nema(matrix_size=matrix_size, voxel_size_mm=voxel_size_mm, nema_dict=base_config)
    z1, y1, x1 = get_sphere_center_mm(act1, voxel_size_mm, matrix_size)
    
    # Test y=-50
    config_y = base_config.copy()
    config_y['sphere_dict'] = base_config['sphere_dict'].copy()
    config_y['sphere_dict']['ring_y'] = -50
    
    act2, _ = create_nema(matrix_size=matrix_size, voxel_size_mm=voxel_size_mm, nema_dict=config_y)
    z2, y2, x2 = get_sphere_center_mm(act2, voxel_size_mm, matrix_size)
    
    # Y should change by -50mm, Z and X should stay the same
    assert abs(y2 - y1 - (-50)) < 3, f"ring_y offset failed: expected -50mm, got {y2-y1:.1f}mm"
    assert abs(z2 - z1) < 3, f"Z position changed when it shouldn't: {z1:.1f} -> {z2:.1f}"
    assert abs(x2 - x1) < 3, f"X position changed when it shouldn't: {x1:.1f} -> {x2:.1f}"


def test_ring_x_positioning():
    """Test that ring_x controls the x-position of the ring center."""
    base_config = {
        'mu_values': {
            'perspex_mu_value': 0.1,
            'fill_mu_value': 0.096,
            'lung_mu_value': 0.029
        },
        'activity_concentration_background': 0,
        'include_lung_insert': False,
        'sphere_dict': {
            'ring_R': 57,
            'ring_z': 0,
            'ring_x': 0,
            'ring_y': 0,
            'spheres': {
                'diametre_mm': [10],
                'angle_loc': [90],  # 90 degrees = +Y direction
                'act_conc_MBq_ml': [0.4]
            }
        }
    }
    
    matrix_size = (128, 128, 128)
    voxel_size_mm = (2, 2, 2)
    
    # Test default x=0
    act1, _ = create_nema(matrix_size=matrix_size, voxel_size_mm=voxel_size_mm, nema_dict=base_config)
    z1, y1, x1 = get_sphere_center_mm(act1, voxel_size_mm, matrix_size)
    
    # Test x=30
    config_x = base_config.copy()
    config_x['sphere_dict'] = base_config['sphere_dict'].copy()
    config_x['sphere_dict']['ring_x'] = 30
    
    act2, _ = create_nema(matrix_size=matrix_size, voxel_size_mm=voxel_size_mm, nema_dict=config_x)
    z2, y2, x2 = get_sphere_center_mm(act2, voxel_size_mm, matrix_size)
    
    # X should change by 30mm, Z and Y should stay the same
    assert abs(x2 - x1 - 30) < 3, f"ring_x offset failed: expected 30mm, got {x2-x1:.1f}mm"
    assert abs(z2 - z1) < 3, f"Z position changed when it shouldn't: {z1:.1f} -> {z2:.1f}"
    assert abs(y2 - y1) < 3, f"Y position changed when it shouldn't: {y1:.1f} -> {y2:.1f}"


def test_ring_xy_combined():
    """Test that ring_x and ring_y work together correctly."""
    base_config = {
        'mu_values': {
            'perspex_mu_value': 0.1,
            'fill_mu_value': 0.096,
            'lung_mu_value': 0.029
        },
        'activity_concentration_background': 0,
        'include_lung_insert': False,
        'sphere_dict': {
            'ring_R': 57,
            'ring_z': 0,
            'ring_x': 0,
            'ring_y': 0,
            'spheres': {
                'diametre_mm': [10],
                'angle_loc': [0],
                'act_conc_MBq_ml': [0.4]
            }
        }
    }
    
    matrix_size = (128, 128, 128)
    voxel_size_mm = (2, 2, 2)
    
    # Test with both offsets
    config_xy = base_config.copy()
    config_xy['sphere_dict'] = base_config['sphere_dict'].copy()
    config_xy['sphere_dict']['ring_x'] = 20
    config_xy['sphere_dict']['ring_y'] = -30
    
    act, _ = create_nema(matrix_size=matrix_size, voxel_size_mm=voxel_size_mm, nema_dict=config_xy)
    z, y, x = get_sphere_center_mm(act, voxel_size_mm, matrix_size)
    
    # At angle 0, sphere should be at (ring_x + ring_R, ring_y)
    expected_x = 20 + 57
    expected_y = -30
    
    assert abs(x - expected_x) < 3, f"Expected x={expected_x}mm, got {x:.1f}mm"
    assert abs(y - expected_y) < 3, f"Expected y={expected_y}mm, got {y:.1f}mm"
