"""Tests for the interactive viewer module."""

import pytest
import numpy as np
import numpy.ma as ma
from unittest.mock import Mock, MagicMock, patch

# Import PyQtGraph and Qt
try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False

# Skip all tests if PyQtGraph is not available
pytestmark = pytest.mark.skipif(
    not PYQTGRAPH_AVAILABLE,
    reason="PyQtGraph not installed"
)

if PYQTGRAPH_AVAILABLE:
    from pypillometry.viewer import view, ViewerWindow
    from pypillometry.viewer.controls import ControlPanel
    from pypillometry.viewer.navigation import NavigationHandler
    from pypillometry.viewer.region_selector import RegionSelector
    from pypillometry.viewer import visualization
    from pypillometry.intervals import Intervals


# Create a mock EyeData object for testing
class MockEyeData:
    """Mock EyeData object for testing."""
    
    def __init__(self, n_samples=1000, has_events=True, has_masks=False):
        """Create mock eye data.
        
        Parameters
        ----------
        n_samples : int
            Number of samples
        has_events : bool
            Whether to include events
        has_masks : bool
            Whether to mask some data
        """
        self.tx = np.linspace(0, 10000, n_samples)  # 10 seconds at 100 Hz
        self.name = "test_data"
        
        # Create data
        data = {}
        data['left_pupil'] = np.random.randn(n_samples) * 100 + 800
        data['right_pupil'] = np.random.randn(n_samples) * 100 + 800
        data['left_x'] = np.random.randn(n_samples) * 50 + 500
        data['left_y'] = np.random.randn(n_samples) * 50 + 400
        data['right_x'] = np.random.randn(n_samples) * 50 + 500
        data['right_y'] = np.random.randn(n_samples) * 50 + 400
        
        # Add masks if requested
        if has_masks:
            for key in data:
                mask = np.zeros(n_samples, dtype=bool)
                mask[100:150] = True  # Mask some samples
                mask[500:550] = True
                data[key] = ma.array(data[key], mask=mask)
        
        self._data = data
        
        # Add events if requested
        if has_events:
            self.event_onsets = np.array([1000, 3000, 5000, 7000, 9000])
            self.event_labels = ['event1', 'event2', 'event3', 'event4', 'event5']
        else:
            self.event_onsets = None
            self.event_labels = None
    
    def __getitem__(self, key):
        return self._data.get(key)
    
    def __contains__(self, key):
        return key in self._data


@pytest.fixture
def mock_eyedata():
    """Create mock eye data for testing."""
    return MockEyeData(n_samples=1000, has_events=True, has_masks=True)


@pytest.fixture
def mock_eyedata_no_events():
    """Create mock eye data without events."""
    return MockEyeData(n_samples=1000, has_events=False, has_masks=False)


@pytest.fixture
def qapp():
    """Create QApplication for tests."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class TestVisualization:
    """Tests for visualization helper functions."""
    
    def test_get_color_for_modality(self):
        """Test color assignment for modalities."""
        assert visualization.get_color_for_modality('left_pupil') == '#0000FF'
        assert visualization.get_color_for_modality('right_pupil') == '#FF0000'
        assert visualization.get_color_for_modality('unknown') == '#7f7f7f'
    
    def test_get_plot_label(self):
        """Test label generation for modalities."""
        assert visualization.get_plot_label('left_pupil') == 'Left Pupil Size'
        assert visualization.get_plot_label('right_x') == 'Right Gaze X'
        assert visualization.get_plot_label('custom_field') == 'Custom Field'


class TestRegionSelector:
    """Tests for RegionSelector class."""
    
    def test_initialization(self, qapp):
        """Test RegionSelector initialization."""
        plot = pg.PlotWidget()
        selector = RegionSelector(plot.getPlotItem())
        
        assert selector.regions == []
        assert selector.callback is None
    
    def test_add_region(self, qapp):
        """Test adding a region."""
        plot = pg.PlotWidget()
        selector = RegionSelector(plot.getPlotItem())
        
        region = selector.add_region(100, 200)
        
        assert len(selector.regions) == 1
        assert region in selector.regions
        start, end = region.getRegion()
        assert start == 100
        assert end == 200
    
    def test_remove_region(self, qapp):
        """Test removing a region."""
        plot = pg.PlotWidget()
        selector = RegionSelector(plot.getPlotItem())
        
        region = selector.add_region(100, 200)
        selector.remove_region(region)
        
        assert len(selector.regions) == 0
    
    def test_clear_regions(self, qapp):
        """Test clearing all regions."""
        plot = pg.PlotWidget()
        selector = RegionSelector(plot.getPlotItem())
        
        selector.add_region(100, 200)
        selector.add_region(300, 400)
        selector.clear_regions()
        
        assert len(selector.regions) == 0
    
    def test_get_intervals(self, qapp):
        """Test getting Intervals from regions."""
        plot = pg.PlotWidget()
        selector = RegionSelector(plot.getPlotItem())
        
        selector.add_region(100, 200)
        selector.add_region(300, 400)
        
        intervals = selector.get_intervals()
        
        assert isinstance(intervals, Intervals)
        assert len(intervals) == 2
        assert intervals.units == 'ms'
    
    def test_get_intervals_empty(self, qapp):
        """Test getting Intervals when no regions."""
        plot = pg.PlotWidget()
        selector = RegionSelector(plot.getPlotItem())
        
        intervals = selector.get_intervals()
        assert intervals is None
    
    def test_set_intervals(self, qapp):
        """Test setting regions from Intervals."""
        plot = pg.PlotWidget()
        selector = RegionSelector(plot.getPlotItem())
        
        intervals = Intervals([(100, 200), (300, 400)], units='ms')
        selector.set_intervals(intervals)
        
        assert len(selector.regions) == 2
    
    def test_callback_called(self, qapp):
        """Test that callback is called on region changes."""
        plot = pg.PlotWidget()
        callback = Mock()
        selector = RegionSelector(plot.getPlotItem(), callback=callback)
        
        selector.add_region(100, 200)
        
        # Callback should be called
        assert callback.call_count >= 1


class TestNavigationHandler:
    """Tests for NavigationHandler class."""
    
    @pytest.mark.skip(reason="Qt object lifecycle issues in test environment")
    def test_initialization(self, qapp):
        """Test NavigationHandler initialization."""
        plot = pg.PlotWidget().getPlotItem()
        nav = NavigationHandler([plot])
        
        assert nav.primary_plot == plot
        assert len(nav.plot_widgets) == 1
    
    @pytest.mark.skip(reason="Qt object lifecycle issues in test environment")
    def test_handle_key_left(self, qapp):
        """Test left arrow key handling."""
        plot = pg.PlotWidget().getPlotItem()
        nav = NavigationHandler([plot])
        
        # Set initial range
        plot.setXRange(0, 1000)
        
        # Create key event for left arrow
        event = MagicMock()
        event.key.return_value = QtCore.Qt.Key_Left
        event.modifiers.return_value = QtCore.Qt.NoModifier
        
        handled = nav.handle_key_press(event)
        assert handled
    
    @pytest.mark.skip(reason="Qt object lifecycle issues in test environment")
    def test_handle_key_right(self, qapp):
        """Test right arrow key handling."""
        plot = pg.PlotWidget().getPlotItem()
        nav = NavigationHandler([plot])
        
        plot.setXRange(0, 1000)
        
        event = MagicMock()
        event.key.return_value = QtCore.Qt.Key_Right
        event.modifiers.return_value = QtCore.Qt.NoModifier
        
        handled = nav.handle_key_press(event)
        assert handled
    
    @pytest.mark.skip(reason="Qt object lifecycle issues in test environment")
    def test_handle_key_space(self, qapp):
        """Test space key (reset view)."""
        plot = pg.PlotWidget().getPlotItem()
        nav = NavigationHandler([plot])
        
        event = MagicMock()
        event.key.return_value = QtCore.Qt.Key_Space
        event.modifiers.return_value = QtCore.Qt.NoModifier
        
        handled = nav.handle_key_press(event)
        assert handled


class TestControlPanel:
    """Tests for ControlPanel class."""
    
    def test_initialization(self, qapp):
        """Test ControlPanel initialization."""
        modalities = ['left_pupil', 'right_pupil', 'left_x']
        panel = ControlPanel(modalities, has_events=True)
        
        assert 'left' in panel.available_eyes
        assert 'right' in panel.available_eyes
        assert 'pupil' in panel.available_variables
        assert 'x' in panel.available_variables
        assert panel.has_events
        assert all(cb.isChecked() for cb in panel.eye_checkboxes.values())
        assert all(cb.isChecked() for cb in panel.variable_checkboxes.values())
    
    def test_get_mask_mode(self, qapp):
        """Test getting mask mode."""
        modalities = ['left_pupil']
        panel = ControlPanel(modalities)
        
        assert panel.get_mask_mode() == 'shaded'
        
        panel.mask_gaps_radio.setChecked(True)
        assert panel.get_mask_mode() == 'gaps'
    
    def test_get_modality_states(self, qapp):
        """Test getting modality states."""
        modalities = ['left_pupil', 'right_pupil']
        panel = ControlPanel(modalities)
        
        states = panel.get_modality_states()
        assert states['left_pupil'] is True
        assert states['right_pupil'] is True
        
        # Uncheck left eye
        panel.eye_checkboxes['left'].setChecked(False)
        states = panel.get_modality_states()
        assert states['left_pupil'] is False
        assert states['right_pupil'] is True
    
    def test_toggle_eye(self, qapp):
        """Test toggling an eye on/off."""
        modalities = ['left_pupil', 'right_pupil', 'left_x']
        panel = ControlPanel(modalities)
        
        # Uncheck left eye
        panel.eye_checkboxes['left'].setChecked(False)
        
        states = panel.get_modality_states()
        assert states['left_pupil'] is False
        assert states['left_x'] is False
        assert states['right_pupil'] is True
    
    def test_toggle_variable(self, qapp):
        """Test toggling a variable on/off."""
        modalities = ['left_pupil', 'right_pupil', 'left_x']
        panel = ControlPanel(modalities)
        
        # Uncheck pupil variable
        panel.variable_checkboxes['pupil'].setChecked(False)
        
        states = panel.get_modality_states()
        assert states['left_pupil'] is False
        assert states['right_pupil'] is False
        assert states['left_x'] is True


class TestViewerWindow:
    """Tests for ViewerWindow class."""
    
    def test_initialization_separate_plots(self, qapp, mock_eyedata):
        """Test ViewerWindow initialization with separate plots."""
        window = ViewerWindow(mock_eyedata, separate_plots=True)
        
        assert window.separate_plots is True
        # Now modalities are grouped by type (pupil, x, y)
        assert 'pupil' in window.available_modalities or 'x' in window.available_modalities
        assert len(window.plot_widgets) <= 3  # Max 3 plots
        assert window.has_events is True
    
    def test_initialization_single_plot(self, qapp, mock_eyedata):
        """Test ViewerWindow initialization with single plot."""
        window = ViewerWindow(mock_eyedata, separate_plots=False)
        
        assert window.separate_plots is False
        assert len(window.plot_widgets) == 1
    
    def test_detect_modalities(self, qapp, mock_eyedata):
        """Test modality detection."""
        window = ViewerWindow(mock_eyedata, separate_plots=True)
        
        modalities = window._detect_modalities()
        # Now returns grouped dictionary
        assert isinstance(modalities, dict)
        assert 'pupil' in modalities
        assert 'left_pupil' in modalities['pupil']
        assert 'right_pupil' in modalities['pupil']
    
    def test_detect_modalities_no_events(self, qapp, mock_eyedata_no_events):
        """Test modality detection without events."""
        window = ViewerWindow(mock_eyedata_no_events, separate_plots=True)
        
        assert window.has_events is False
    
    def test_get_intervals_none(self, qapp, mock_eyedata):
        """Test getting intervals when none selected."""
        window = ViewerWindow(mock_eyedata, separate_plots=True)
        
        intervals = window.get_intervals()
        assert intervals is None
    
    def test_get_intervals_with_selection(self, qapp, mock_eyedata):
        """Test getting intervals with selection."""
        window = ViewerWindow(mock_eyedata, separate_plots=True)
        
        # Add a region
        window.region_selector.add_region(1000, 2000)
        
        intervals = window.get_intervals()
        assert intervals is not None
        assert len(intervals) == 1
    
    def test_modality_toggle(self, qapp, mock_eyedata):
        """Test toggling modality visibility."""
        window = ViewerWindow(mock_eyedata, separate_plots=True)
        
        # Toggle off left_pupil
        window._on_modality_toggled('left_pupil', False)
        
        # Check that curve is not visible
        assert not window.plot_curves['left_pupil'].isVisible()
        
        # Toggle back on
        window._on_modality_toggled('left_pupil', True)
        assert window.plot_curves['left_pupil'].isVisible()
    
    def test_events_toggle(self, qapp, mock_eyedata):
        """Test toggling event markers."""
        window = ViewerWindow(mock_eyedata, separate_plots=True)
        
        # Events should be hidden initially (new default)
        assert all(not marker.isVisible() for marker in window.event_markers)
        
        # Toggle on
        window._on_events_toggled(True)
        assert all(marker.isVisible() for marker in window.event_markers)
        
        # Toggle back off
        window._on_events_toggled(False)
        assert all(not marker.isVisible() for marker in window.event_markers)
    
    def test_accept_cancel(self, qapp, mock_eyedata):
        """Test accept and cancel actions."""
        window = ViewerWindow(mock_eyedata, separate_plots=True)
        
        # Add a region
        window.region_selector.add_region(1000, 2000)
        
        # Test accept
        window._on_accept()
        assert window.accepted is True
        assert window.selected_intervals is not None
        
        # Create new window for cancel test
        window2 = ViewerWindow(mock_eyedata, separate_plots=True)
        window2._on_cancel()
        assert window2.accepted is False
        assert window2.selected_intervals is None


class TestViewAPI:
    """Tests for the main view() API function."""
    
    @patch('pypillometry.viewer.QtWidgets.QApplication.exec_')
    def test_view_blocking_mode(self, mock_exec, qapp, mock_eyedata):
        """Test view() in blocking mode."""
        # Mock exec_ to return immediately
        mock_exec.return_value = 0
        
        # This will create the window but not actually show it in test
        # We can't fully test blocking behavior in unit tests
        # Just verify it doesn't crash
        with patch.object(ViewerWindow, 'show'):
            result = view(mock_eyedata, separate_plots=True, callback=None)
    
    def test_view_non_blocking_mode(self, qapp, mock_eyedata):
        """Test view() in non-blocking mode."""
        callback = Mock()
        
        with patch.object(ViewerWindow, 'show'):
            result = view(mock_eyedata, separate_plots=True, callback=callback)
        
        # Should return ViewerWindow instance
        assert isinstance(result, ViewerWindow)
    
    def test_view_separate_plots_true(self, qapp, mock_eyedata):
        """Test view() with separate_plots=True."""
        with patch.object(ViewerWindow, 'show'):
            result = view(mock_eyedata, separate_plots=True, callback=Mock())
        
        assert isinstance(result, ViewerWindow)
        assert result.separate_plots is True
    
    def test_view_separate_plots_false(self, qapp, mock_eyedata):
        """Test view() with separate_plots=False."""
        with patch.object(ViewerWindow, 'show'):
            result = view(mock_eyedata, separate_plots=False, callback=Mock())
        
        assert isinstance(result, ViewerWindow)
        assert result.separate_plots is False


class TestIntegration:
    """Integration tests for the viewer."""
    
    def test_full_workflow(self, qapp, mock_eyedata):
        """Test complete workflow: create window, add region, get intervals."""
        window = ViewerWindow(mock_eyedata, separate_plots=True)
        
        # Add regions
        window.region_selector.add_region(1000, 2000)
        window.region_selector.add_region(3000, 4000)
        
        # Get intervals
        intervals = window.get_intervals()
        
        assert intervals is not None
        assert len(intervals) == 2
        assert intervals.units == 'ms'
        
        # Clear regions
        window.region_selector.clear_regions()
        intervals = window.get_intervals()
        assert intervals is None
    
    def test_masked_data_handling(self, qapp):
        """Test handling of masked data."""
        # Create data with masks
        data = MockEyeData(n_samples=500, has_masks=True)
        window = ViewerWindow(data, separate_plots=True)
        
        # Should have mask regions
        assert len(window.mask_regions) > 0
    
    def test_no_events_handling(self, qapp, mock_eyedata_no_events):
        """Test handling when no events are present."""
        window = ViewerWindow(mock_eyedata_no_events, separate_plots=True)
        
        assert window.has_events is False
        assert len(window.event_markers) == 0

