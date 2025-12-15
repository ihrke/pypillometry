#!/usr/bin/env python
"""Test script for flexible viewer API.

Run with: python test_viewer_flexible.py
"""

import numpy as np
import sys

# Add parent directory to path if needed
sys.path.insert(0, '.')

import pypillometry as pp

def test_normalization():
    """Test the input normalization function."""
    from pypillometry.viewer import _normalize_input, _validate_array_lengths
    
    print("Testing input normalization...")
    
    # Test single array
    x = np.random.randn(100)
    mode, spec, time = _normalize_input(x)
    assert mode == 'arrays', f"Expected 'arrays', got {mode}"
    assert 'Signal' in spec, f"Expected 'Signal' key, got {spec.keys()}"
    assert len(spec['Signal']) == 1, f"Expected 1 array, got {len(spec['Signal'])}"
    assert len(time) == 100, f"Expected 100 time points, got {len(time)}"
    print("  Single array: OK")
    
    # Test list of arrays
    y = np.random.randn(100)
    mode, spec, time = _normalize_input([x, y])
    assert mode == 'arrays', f"Expected 'arrays', got {mode}"
    assert len(spec['Signal']) == 2, f"Expected 2 arrays, got {len(spec['Signal'])}"
    print("  List of arrays: OK")
    
    # Test dict of arrays
    mode, spec, time = _normalize_input({"upper": [x, y], "lower": x + y})
    assert mode == 'arrays', f"Expected 'arrays', got {mode}"
    assert 'upper' in spec, f"Expected 'upper' key"
    assert 'lower' in spec, f"Expected 'lower' key"
    assert len(spec['upper']) == 2, f"Expected 2 arrays in upper"
    assert len(spec['lower']) == 1, f"Expected 1 array in lower"
    print("  Dict of arrays: OK")
    
    # Test with custom time
    t = np.linspace(0, 10, 100)
    mode, spec, time = _normalize_input(x, time=t)
    assert np.allclose(time, t.astype(np.float32)), "Time vector not preserved"
    print("  Custom time: OK")
    
    # Test validation
    try:
        _validate_array_lengths([np.zeros(10), np.zeros(20)])
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  Length validation: OK")
    
    # Test masked array
    masked = np.ma.array([1, 2, 3, 4, 5], mask=[0, 0, 1, 0, 0])
    mode, spec, time = _normalize_input(masked)
    assert mode == 'arrays', f"Expected 'arrays', got {mode}"
    print("  Masked array: OK")
    
    print("All normalization tests passed!\n")


def test_viewer_single_array():
    """Test viewing a single array (visual test)."""
    print("Testing single array view...")
    print("  A window should open with one plot.")
    print("  Press 'q' to close and continue.\n")
    
    x = np.sin(np.linspace(0, 4 * np.pi, 1000)) + np.random.randn(1000) * 0.1
    pp.view(x)
    print("  Single array view: OK\n")


def test_viewer_list():
    """Test viewing list of arrays (visual test)."""
    print("Testing list of arrays view...")
    print("  A window should open with one plot, two signals in different colors.")
    print("  Press 'q' to close and continue.\n")
    
    t = np.linspace(0, 10, 1000)
    x = np.sin(2 * np.pi * t)
    y = np.cos(2 * np.pi * t)
    pp.view([x, y], time=t)
    print("  List of arrays view: OK\n")


def test_viewer_dict():
    """Test viewing dict of arrays (visual test)."""
    print("Testing dict of arrays view...")
    print("  A window should open with two plots labeled 'Upper' and 'Lower'.")
    print("  Upper should have two signals, Lower should have one.")
    print("  Press 'q' to close and continue.\n")
    
    t = np.linspace(0, 10, 1000)
    x = np.sin(2 * np.pi * t)
    y = np.cos(2 * np.pi * t)
    z = x + y
    
    pp.view({"Upper": [x, y], "Lower": z}, time=t)
    print("  Dict of arrays view: OK\n")


def test_viewer_eyedata_variables():
    """Test EyeData with variables filter (visual test)."""
    print("Testing EyeData with variables filter...")
    print("  Loading example data...")
    
    try:
        data = pp.example_data.get_medium_example()
        print("  A window should open with ONLY pupil plots (no gaze x/y).")
        print("  Press 'q' to close and continue.\n")
        pp.view(data, variables=['pupil'])
        print("  EyeData variables filter: OK\n")
    except Exception as e:
        print(f"  Skipping EyeData test (couldn't load example data): {e}\n")


if __name__ == '__main__':
    # Run unit tests first
    test_normalization()
    
    # Run visual tests
    print("=" * 50)
    print("VISUAL TESTS - Close each window to continue")
    print("=" * 50)
    print()
    
    test_viewer_single_array()
    test_viewer_list()
    test_viewer_dict()
    test_viewer_eyedata_variables()
    
    print("=" * 50)
    print("All tests completed!")
    print("=" * 50)
