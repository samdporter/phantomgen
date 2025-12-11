import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from phantomgen.core import _build_parser


def test_build_parser_defaults(tmp_path, monkeypatch):
    """Test CLI with default arguments."""
    monkeypatch.chdir(tmp_path)
    
    with patch.object(sys, 'argv', ['phantomgen']):
        _build_parser()
    
    # Check that default output files were created
    assert Path("activity.npy").exists()
    assert Path("ctmu.npy").exists()
    
    # Load and verify shapes
    act = np.load("activity.npy")
    ct = np.load("ctmu.npy")
    assert act.shape == (256, 256, 256)
    assert ct.shape == (256, 256, 256)
    assert act.dtype == np.float32
    assert ct.dtype == np.float32


def test_build_parser_custom_matrix_size(tmp_path, monkeypatch):
    """Test CLI with custom matrix size."""
    monkeypatch.chdir(tmp_path)
    
    with patch.object(sys, 'argv', [
        'phantomgen',
        '--z', '64',
        '--y', '64',
        '--x', '64',
        '--out-act', 'act_custom.npy',
        '--out-ct', 'ct_custom.npy'
    ]):
        _build_parser()
    
    act = np.load("act_custom.npy")
    ct = np.load("ct_custom.npy")
    assert act.shape == (64, 64, 64)
    assert ct.shape == (64, 64, 64)


def test_build_parser_custom_voxel_size(tmp_path, monkeypatch):
    """Test CLI with custom voxel size."""
    monkeypatch.chdir(tmp_path)
    
    with patch.object(sys, 'argv', [
        'phantomgen',
        '--z', '32',
        '--y', '32',
        '--x', '32',
        '--voxel', '4.0', '4.0', '4.0',
        '--out-act', 'act_vox.npy',
        '--out-ct', 'ct_vox.npy'
    ]):
        _build_parser()
    
    act = np.load("act_vox.npy")
    ct = np.load("ct_vox.npy")
    assert act.shape == (32, 32, 32)


def test_build_parser_earl_preset(tmp_path, monkeypatch):
    """Test CLI with EARL preset."""
    monkeypatch.chdir(tmp_path)
    
    with patch.object(sys, 'argv', [
        'phantomgen',
        '--preset', 'earl',
        '--z', '32',
        '--y', '32',
        '--x', '32',
        '--out-act', 'act_earl.npy',
        '--out-ct', 'ct_earl.npy'
    ]):
        _build_parser()
    
    assert Path("act_earl.npy").exists()
    assert Path("ct_earl.npy").exists()


def test_build_parser_pet_preset(tmp_path, monkeypatch):
    """Test CLI with PET preset."""
    monkeypatch.chdir(tmp_path)
    
    with patch.object(sys, 'argv', [
        'phantomgen',
        '--preset', 'pet',
        '--z', '32',
        '--y', '32',
        '--x', '32',
        '--out-act', 'act_pet.npy',
        '--out-ct', 'ct_pet.npy'
    ]):
        _build_parser()
    
    assert Path("act_pet.npy").exists()
    assert Path("ct_pet.npy").exists()


def test_build_parser_with_offset(tmp_path, monkeypatch):
    """Test CLI with center offset."""
    monkeypatch.chdir(tmp_path)
    
    with patch.object(sys, 'argv', [
        'phantomgen',
        '--z', '32',
        '--y', '32',
        '--x', '32',
        '--offset', '10.0', '5.0', '-5.0',
        '--out-act', 'act_offset.npy',
        '--out-ct', 'ct_offset.npy'
    ]):
        _build_parser()
    
    assert Path("act_offset.npy").exists()
    assert Path("ct_offset.npy").exists()


def test_build_parser_supersample_isotropic(tmp_path, monkeypatch):
    """Test CLI with isotropic supersampling."""
    monkeypatch.chdir(tmp_path)
    
    with patch.object(sys, 'argv', [
        'phantomgen',
        '--z', '32',
        '--y', '32',
        '--x', '32',
        '--supersample', '2',
        '--out-act', 'act_ss.npy',
        '--out-ct', 'ct_ss.npy'
    ]):
        _build_parser()
    
    act = np.load("act_ss.npy")
    ct = np.load("ct_ss.npy")
    assert act.shape == (32, 32, 32)
    assert ct.shape == (32, 32, 32)


def test_build_parser_supersample_anisotropic(tmp_path, monkeypatch):
    """Test CLI with anisotropic supersampling."""
    monkeypatch.chdir(tmp_path)
    
    with patch.object(sys, 'argv', [
        'phantomgen',
        '--z', '30',
        '--y', '30',
        '--x', '30',
        '--supersample', '2', '3', '1',
        '--out-act', 'act_aniso.npy',
        '--out-ct', 'ct_aniso.npy'
    ]):
        _build_parser()
    
    act = np.load("act_aniso.npy")
    ct = np.load("ct_aniso.npy")
    assert act.shape == (30, 30, 30)


def test_build_parser_supersample_invalid_count(tmp_path, monkeypatch):
    """Test CLI with invalid number of supersample values."""
    monkeypatch.chdir(tmp_path)
    
    with patch.object(sys, 'argv', [
        'phantomgen',
        '--supersample', '2', '3'
    ]):
        with pytest.raises(SystemExit):
            _build_parser()


def test_build_parser_supersample_invalid_value(tmp_path, monkeypatch):
    """Test CLI with invalid supersample value."""
    monkeypatch.chdir(tmp_path)
    
    with patch.object(sys, 'argv', [
        'phantomgen',
        '--supersample', '0'
    ]):
        with pytest.raises(SystemExit):
            _build_parser()
