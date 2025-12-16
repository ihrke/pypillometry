#!/usr/bin/env python
"""Generate screenshots for viewer documentation.

This script creates screenshots of pp.view() and pp.tweak() examples
by directly instantiating the canvas classes (bypassing the blocking event loop).

Usage:
    python generate_viewer_screenshots.py

Screenshots are saved to docs/_static/
"""

import sys
import os

# Add parent directories to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
docs_dir = os.path.dirname(script_dir)
project_dir = os.path.dirname(docs_dir)
sys.path.insert(0, project_dir)

import numpy as np
import vispy
vispy.use('pyqt6')  # or pyqt5

from vispy import app
from imageio import imwrite

import pypillometry as pp
from pypillometry.viewer.canvas import ViewerCanvas
from pypillometry.viewer.tweak import TweakCanvas, TweakViewer

# Import Qt for window screenshots
try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt, QTimer
except ImportError:
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt, QTimer

# Output directory
OUTPUT_DIR = os.path.join(docs_dir, '_static')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_canvas_screenshot(canvas, filename, process_events=5):
    """Render VisPy canvas and save screenshot.
    
    Parameters
    ----------
    canvas : SceneCanvas
        VisPy canvas to render.
    filename : str
        Output filename (will be saved in OUTPUT_DIR).
    process_events : int
        Number of times to process events before rendering.
    """
    canvas.show()
    
    # Process events multiple times to ensure full rendering
    for _ in range(process_events):
        canvas.app.process_events()
    
    # Force a draw
    canvas.update()
    canvas.app.process_events()
    
    # Render to image
    image = canvas.render()
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, filename)
    imwrite(output_path, image)
    print(f"Saved: {output_path}")
    
    canvas.close()


def save_window_screenshot(window, filename, process_events=10):
    """Capture full Qt window screenshot including decorations.
    
    Parameters
    ----------
    window : QWidget
        Qt window to capture.
    filename : str
        Output filename (will be saved in OUTPUT_DIR).
    process_events : int
        Number of times to process events before capturing.
    """
    qt_app = QApplication.instance()
    
    window.show()
    
    # Process events multiple times to ensure full rendering
    for _ in range(process_events):
        qt_app.processEvents()
    
    # Give the window time to fully render
    import time
    time.sleep(0.1)
    qt_app.processEvents()
    
    # Grab the window contents (including all widgets)
    pixmap = window.grab()
    
    # Convert to numpy array via QImage
    qimage = pixmap.toImage()
    
    # Get image dimensions
    width = qimage.width()
    height = qimage.height()
    
    # Convert QImage to numpy array
    ptr = qimage.bits()
    if hasattr(ptr, 'setsize'):  # PyQt5
        ptr.setsize(height * width * 4)
    arr = np.array(ptr, dtype=np.uint8).reshape((height, width, 4))
    
    # QImage is BGRA, convert to RGBA
    image = arr[:, :, [2, 1, 0, 3]].copy()
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, filename)
    imwrite(output_path, image)
    print(f"Saved: {output_path}")
    
    window.close()


def screenshot_view_eyedata():
    """Screenshot: Basic EyeData viewing."""
    print("\n=== Generating: viewer_eyedata.png ===")
    
    # Load example data
    eyedata = pp.get_example_data("rlmw_002_short")
    time_vec = eyedata.tx.astype(np.float32) * 0.001  # ms to seconds
    
    canvas = ViewerCanvas(
        eyedata,
        mode='eyedata',
        time_seconds=time_vec,
        variables=None,  # Show all
    )
    
    save_canvas_screenshot(canvas, 'viewer_eyedata.png')


def screenshot_view_arrays():
    """Screenshot: Viewing numpy arrays with dict layout."""
    print("\n=== Generating: viewer_arrays.png ===")
    
    # Create synthetic data
    np.random.seed(42)
    t = np.linspace(0, 10, 1000, dtype=np.float32)
    signal1 = np.sin(2 * np.pi * 0.5 * t) + np.random.randn(1000).astype(np.float32) * 0.1
    signal2 = np.cos(2 * np.pi * 0.3 * t) + np.random.randn(1000).astype(np.float32) * 0.1
    
    plot_spec = {
        'Signal A': [signal1],
        'Signal B': [signal2],
    }
    
    canvas = ViewerCanvas(
        plot_spec,
        mode='arrays',
        time_seconds=t,
    )
    
    save_canvas_screenshot(canvas, 'viewer_arrays.png')


def screenshot_view_pupil_only():
    """Screenshot: EyeData with only pupil variables."""
    print("\n=== Generating: viewer_pupil_only.png ===")
    
    eyedata = pp.get_example_data("rlmw_002_short")
    time_vec = eyedata.tx.astype(np.float32) * 0.001
    
    canvas = ViewerCanvas(
        eyedata,
        mode='eyedata',
        time_seconds=time_vec,
        variables=['pupil'],  # Only pupil
    )
    
    save_canvas_screenshot(canvas, 'viewer_pupil_only.png')


def screenshot_view_with_overlay():
    """Screenshot: EyeData with overlay (smoothed signal)."""
    print("\n=== Generating: viewer_overlay.png ===")
    
    eyedata = pp.get_example_data("rlmw_002_short")
    time_vec = eyedata.tx.astype(np.float32) * 0.001
    
    # Create smoothed version for overlay
    pupil = np.array(eyedata['left_pupil'])
    # Simple moving average
    window = 50
    kernel = np.ones(window) / window
    smoothed = np.convolve(pupil, kernel, mode='same').astype(np.float32)
    
    canvas = ViewerCanvas(
        eyedata,
        mode='eyedata',
        time_seconds=time_vec,
        variables=['pupil'],
        overlays={'pupil': {'smoothed': smoothed}},
    )
    
    save_canvas_screenshot(canvas, 'viewer_overlay.png')


def screenshot_view_extra_plots():
    """Screenshot: EyeData with extra subplot."""
    print("\n=== Generating: viewer_extra_plots.png ===")
    
    eyedata = pp.get_example_data("rlmw_002_short")
    time_vec = eyedata.tx.astype(np.float32) * 0.001
    
    # Compute velocity as extra plot
    pupil = np.array(eyedata['left_pupil'], dtype=np.float32)
    velocity = np.gradient(pupil).astype(np.float32)
    
    canvas = ViewerCanvas(
        eyedata,
        mode='eyedata',
        time_seconds=time_vec,
        variables=['pupil'],
        extra_plots={'velocity': [velocity]},
    )
    
    save_canvas_screenshot(canvas, 'viewer_extra_plots.png')


def screenshot_tweak_basic():
    """Screenshot: Basic tweak viewer with smoothing function."""
    print("\n=== Generating: tweak_basic.png ===")
    
    # Create noisy signal
    np.random.seed(42)
    t = np.linspace(0, 10, 1000, dtype=np.float32)
    signal = np.sin(2 * np.pi * 0.5 * t) + np.random.randn(1000).astype(np.float32) * 0.3
    
    # Initial smoothing parameters
    params = {'window_size': 20}
    
    # Simple smoothing function result
    window_size = params['window_size']
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(signal, kernel, mode='same').astype(np.float32)
    
    canvas = TweakCanvas(
        time_seconds=t,
        original_data=signal,
        overlay_data={'smoothed': smoothed},
        title='Tweak: smooth'
    )
    
    # Create the full TweakViewer with parameter panel
    def on_change(new_params):
        pass  # No-op for screenshot
    
    viewer = TweakViewer(
        canvas=canvas,
        params=params,
        on_change=on_change,
        title='Tweak: smooth'
    )
    
    save_window_screenshot(viewer.main_window, 'tweak_basic.png')


def screenshot_tweak_eyedata():
    """Screenshot: Tweak viewer with EyeData (lowpass filter)."""
    print("\n=== Generating: tweak_eyedata.png ===")
    
    eyedata = pp.get_example_data("rlmw_002_short")
    time_vec = eyedata.tx.astype(np.float32) * 0.001
    
    # Get original pupil data
    original = np.array(eyedata['left_pupil'], dtype=np.float32)
    
    # Initial parameters
    params = {'cutoff': 4.0}
    
    # Apply lowpass filter
    filtered_data = eyedata.pupil_lowpass_filter(cutoff=params['cutoff'], inplace=False)
    filtered = np.array(filtered_data['left_pupil'], dtype=np.float32)
    
    # Get mask
    mask = None
    try:
        mask = eyedata.data.mask.get('left_pupil')
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
    except (AttributeError, KeyError):
        pass
    
    canvas = TweakCanvas(
        time_seconds=time_vec,
        original_data=original,
        original_mask=mask,
        overlay_data={'filtered': filtered},
        sampling_rate=eyedata.fs,
        title='Tweak: pupil_lowpass_filter'
    )
    
    # Create the full TweakViewer with parameter panel
    def on_change(new_params):
        pass  # No-op for screenshot
    
    viewer = TweakViewer(
        canvas=canvas,
        params=params,
        on_change=on_change,
        title='Tweak: pupil_lowpass_filter'
    )
    
    save_window_screenshot(viewer.main_window, 'tweak_eyedata.png')


def screenshot_tweak_intervals():
    """Screenshot: Tweak viewer showing intervals (e.g., detected blinks)."""
    print("\n=== Generating: tweak_intervals.png ===")
    
    eyedata = pp.get_example_data("rlmw_002_short")
    time_vec = eyedata.tx.astype(np.float32) * 0.001
    
    # Get original pupil data
    original = np.array(eyedata['left_pupil'], dtype=np.float32)
    
    # Initial parameters for blink detection
    params = {'vel_onset': -5.0, 'vel_offset': 5.0}
    
    # Detect blinks to get intervals
    blinks = eyedata.pupil_blinks_detect(
        apply_mask=False, 
        vel_onset=params['vel_onset'],
        vel_offset=params['vel_offset']
    )
    blink_intervals = blinks.get('left_pupil')
    
    # Convert Intervals to list of (start, end) tuples in seconds
    intervals_list = []
    if blink_intervals is not None:
        # Convert to seconds
        if blink_intervals.units != 'sec':
            blink_intervals = blink_intervals.to_units('sec')
        intervals_list = list(blink_intervals.intervals)
    
    canvas = TweakCanvas(
        time_seconds=time_vec,
        original_data=original,
        overlay_intervals={'blinks': intervals_list} if intervals_list else None,
        sampling_rate=eyedata.fs,
        title='Tweak: pupil_blinks_detect'
    )
    
    # Create the full TweakViewer with parameter panel
    def on_change(new_params):
        pass  # No-op for screenshot
    
    viewer = TweakViewer(
        canvas=canvas,
        params=params,
        on_change=on_change,
        title='Tweak: pupil_blinks_detect'
    )
    
    save_window_screenshot(viewer.main_window, 'tweak_intervals.png')


def main():
    """Generate all screenshots."""
    print("Generating viewer documentation screenshots...")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Initialize Qt app
    qt_app = app.use_app().native
    
    # Generate all screenshots
    try:
        # pp.view() examples
        screenshot_view_eyedata()
        screenshot_view_arrays()
        screenshot_view_pupil_only()
        screenshot_view_with_overlay()
        screenshot_view_extra_plots()
        
        # pp.tweak() examples
        screenshot_tweak_basic()
        screenshot_tweak_eyedata()
        screenshot_tweak_intervals()
        
    except Exception as e:
        print(f"Error generating screenshots: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n=== All screenshots generated successfully! ===")
    return 0


if __name__ == '__main__':
    sys.exit(main())

