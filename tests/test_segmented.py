"""
Tests for the SegmentedEyeData class.
"""
import unittest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "..")
import pypillometry as pp
import numpy as np
import numpy.ma as ma
from pypillometry.segmented import SegmentedEyeData, GroupLevelSegmentedData, group_segments
from pypillometry.intervals import Intervals


class TestSegmentedEyeDataCreation(unittest.TestCase):
    """Test SegmentedEyeData creation methods"""
    
    def test_creation_from_masked_array(self):
        """Test creating SegmentedEyeData from masked array"""
        n_timepoints = 100
        n_segments = 10
        tx = np.linspace(-200, 800, n_timepoints)
        data = np.random.randn(n_timepoints, n_segments)
        mask = np.zeros((n_timepoints, n_segments), dtype=bool)
        mask[0:5, 0] = True  # Mask first 5 timepoints of first segment
        
        masked_data = ma.masked_array(data, mask=mask)
        
        seg = SegmentedEyeData.from_masked_array(
            data=masked_data,
            tx=tx,
            variable="left_pupil",
            name="test_segments"
        )
        
        self.assertEqual(seg.name, "test_segments")
        self.assertEqual(seg.variable, "left_pupil")
        self.assertEqual(seg.n_timepoints, n_timepoints)
        self.assertEqual(seg.n_segments, n_segments)
        self.assertEqual(seg.eye, "left")
        self.assertEqual(seg.var, "pupil")
        np.testing.assert_array_equal(seg.tx, tx)
        self.assertTrue(seg.data.mask[0, 0])
        self.assertFalse(seg.data.mask[10, 0])
    
    def test_creation_with_separate_mask(self):
        """Test creating SegmentedEyeData with separate mask array"""
        n_timepoints = 50
        n_segments = 5
        tx = np.linspace(-100, 400, n_timepoints)
        data = np.random.randn(n_timepoints, n_segments)
        mask = np.zeros((n_timepoints, n_segments), dtype=bool)
        mask[:, 2] = True  # Mask entire segment 2
        
        seg = SegmentedEyeData.from_masked_array(
            data=data,
            tx=tx,
            variable="right_x",
            mask=mask
        )
        
        self.assertEqual(seg.var, "x")
        self.assertEqual(seg.eye, "right")
        self.assertTrue(np.all(seg.data.mask[:, 2]))
        self.assertFalse(np.any(seg.data.mask[:, 0]))
    
    def test_creation_from_eyedata(self):
        """Test creating SegmentedEyeData from GenericEyeData via get_segments"""
        # Create PupilData with events
        n_samples = 1000
        fs = 100
        pupil = np.sin(np.linspace(0, 10*np.pi, n_samples)) * 100 + 500
        event_onsets = [1000, 3000, 5000, 7000]  # in ms
        event_labels = ["stim1", "stim2", "stim3", "stim4"]
        
        data = pp.PupilData(
            sampling_rate=fs,
            left_pupil=pupil,
            event_onsets=event_onsets,
            event_labels=event_labels
        )
        
        # Get intervals and segments
        intervals = data.get_intervals("stim", interval=(-100, 200), units="ms")
        segments = data.get_segments(intervals, "left_pupil")
        
        self.assertIsInstance(segments, SegmentedEyeData)
        self.assertEqual(segments.variable, "left_pupil")
        self.assertEqual(segments.n_segments, 4)
        self.assertIsNotNone(segments.sampling_rate)
    
    def test_variable_format_validation(self):
        """Test that variable must be in eye_variable format"""
        tx = np.linspace(-100, 100, 50)
        data = np.random.randn(50, 5)
        
        with self.assertRaises(ValueError):
            SegmentedEyeData.from_masked_array(
                data=data,
                tx=tx,
                variable="pupil"  # Missing eye prefix
            )
    
    def test_variable_not_found_error(self):
        """Test error when variable doesn't exist in eyedata"""
        data = pp.PupilData(
            sampling_rate=100,
            left_pupil=np.random.randn(100),
            event_onsets=[500],
            event_labels=["test"]
        )
        
        intervals = data.get_intervals("test", interval=(-50, 100), units="ms")
        
        with self.assertRaises(KeyError):
            data.get_segments(intervals, "right_pupil")  # Only left exists


class TestSegmentedEyeDataProperties(unittest.TestCase):
    """Test SegmentedEyeData properties"""
    
    def setUp(self):
        """Create test segment data"""
        self.n_timepoints = 100
        self.n_segments = 10
        self.tx = np.linspace(-200, 800, self.n_timepoints)
        data = np.random.randn(self.n_timepoints, self.n_segments)
        
        self.seg = SegmentedEyeData.from_masked_array(
            data=data,
            tx=self.tx,
            variable="left_pupil",
            name="test",
            sampling_rate=500.0
        )
    
    def test_n_timepoints(self):
        """Test n_timepoints property"""
        self.assertEqual(self.seg.n_timepoints, self.n_timepoints)
    
    def test_n_segments(self):
        """Test n_segments property"""
        self.assertEqual(self.seg.n_segments, self.n_segments)
    
    def test_eye_property(self):
        """Test eye property extraction"""
        self.assertEqual(self.seg.eye, "left")
    
    def test_var_property(self):
        """Test var property extraction"""
        self.assertEqual(self.seg.var, "pupil")
    
    def test_window_property(self):
        """Test window property"""
        window = self.seg.window
        self.assertEqual(window[0], -200)
        self.assertEqual(window[1], 800)
    
    def test_summary(self):
        """Test summary method"""
        summary = self.seg.summary()
        
        self.assertEqual(summary["name"], "test")
        self.assertEqual(summary["variable"], "left_pupil")
        self.assertEqual(summary["n_segments"], self.n_segments)
        self.assertEqual(summary["n_timepoints"], self.n_timepoints)
        self.assertEqual(summary["sampling_rate"], 500.0)
        self.assertIn("percent_masked", summary)
        self.assertIn("window", summary)
    
    def test_repr(self):
        """Test string representation"""
        repr_str = repr(self.seg)
        
        self.assertIn("SegmentedEyeData(test)", repr_str)
        self.assertIn("variable", repr_str)
        self.assertIn("n_segments", repr_str)
    
    def test_repr_html(self):
        """Test HTML representation for notebooks"""
        html = self.seg._repr_html_()
        
        self.assertIn("SegmentedEyeData", html)
        self.assertIn("test", html)  # name
        self.assertIn("left_pupil", html)  # variable
        self.assertIn("Segments", html)
        self.assertIn("Time points", html)


class TestSegmentedEyeDataBaselineCorrection(unittest.TestCase):
    """Test baseline correction functionality"""
    
    def test_baseline_correct_single_point(self):
        """Test baseline correction at single time point"""
        n_timepoints = 100
        n_segments = 5
        tx = np.linspace(-200, 800, n_timepoints)
        
        # Create data where each segment has different baseline
        data = np.zeros((n_timepoints, n_segments))
        for i in range(n_segments):
            data[:, i] = i * 10 + np.random.randn(n_timepoints) * 0.1
        
        seg = SegmentedEyeData.from_masked_array(
            data=data,
            tx=tx,
            variable="left_pupil"
        )
        
        # Baseline correct at t=0
        seg_bc = seg.baseline_correct(window=0)
        
        # After correction, values at t=0 should be close to 0
        t0_idx = np.argmin(np.abs(tx))
        for i in range(n_segments):
            self.assertAlmostEqual(seg_bc.data[t0_idx, i], 0, places=1)
    
    def test_baseline_correct_window(self):
        """Test baseline correction with time window"""
        n_timepoints = 100
        n_segments = 5
        tx = np.linspace(-200, 800, n_timepoints)
        
        # Create data with known offsets
        data = np.zeros((n_timepoints, n_segments))
        for i in range(n_segments):
            data[:, i] = (i + 1) * 100  # Constant value per segment
        
        seg = SegmentedEyeData.from_masked_array(
            data=data,
            tx=tx,
            variable="left_pupil"
        )
        
        # Baseline correct using window (-200, 0)
        seg_bc = seg.baseline_correct(window=(-200, 0))
        
        # After correction, all segments should be at 0
        for i in range(n_segments):
            self.assertAlmostEqual(seg_bc.data[:, i].mean(), 0, places=5)
    
    def test_baseline_correct_none(self):
        """Test that None window returns original data"""
        tx = np.linspace(-200, 800, 100)
        data = np.random.randn(100, 5)
        
        seg = SegmentedEyeData.from_masked_array(
            data=data,
            tx=tx,
            variable="left_pupil"
        )
        
        seg_bc = seg.baseline_correct(window=None)
        
        # Should return same object reference since no correction applied
        self.assertIs(seg_bc, seg)
    
    def test_baseline_correct_returns_new_instance(self):
        """Test that baseline correction returns new instance"""
        tx = np.linspace(-200, 800, 100)
        data = np.random.randn(100, 5) + 100
        
        seg = SegmentedEyeData.from_masked_array(
            data=data,
            tx=tx,
            variable="left_pupil"
        )
        original_mean = seg.data.mean()
        
        seg_bc = seg.baseline_correct(window=0)
        
        # Original should be unchanged
        self.assertAlmostEqual(seg.data.mean(), original_mean)
        # New instance should be different
        self.assertIsNot(seg_bc, seg)


class TestSegmentedEyeDataMasking(unittest.TestCase):
    """Test mask handling in SegmentedEyeData"""
    
    def test_mask_preserved_from_eyedata(self):
        """Test that masks from source data are preserved"""
        n_samples = 500
        fs = 100
        pupil = np.random.randn(n_samples) + 500
        pupil[100:120] = np.nan  # Create missing data
        
        data = pp.PupilData(
            sampling_rate=fs,
            left_pupil=pupil,
            event_onsets=[1000, 2000, 3000],
            event_labels=["stim"] * 3
        )
        
        intervals = data.get_intervals("stim", interval=(-50, 150), units="ms")
        segments = data.get_segments(intervals, "left_pupil")
        
        # Check that mask exists
        self.assertIsInstance(segments.data, ma.MaskedArray)
    
    def test_percent_masked_in_summary(self):
        """Test percent_masked calculation in summary"""
        n_timepoints = 100
        n_segments = 10
        tx = np.linspace(-200, 800, n_timepoints)
        data = np.random.randn(n_timepoints, n_segments)
        
        # Mask 10% of data
        mask = np.zeros((n_timepoints, n_segments), dtype=bool)
        mask[0:10, :] = True  # First 10 timepoints masked
        
        seg = SegmentedEyeData.from_masked_array(
            data=data,
            tx=tx,
            variable="left_pupil",
            mask=mask
        )
        
        summary = seg.summary()
        self.assertAlmostEqual(summary["percent_masked"], 10.0)


class TestSegmentedEyeDataPlotting(unittest.TestCase):
    """Test plotting functionality"""
    
    def setUp(self):
        """Create test data"""
        n_timepoints = 100
        n_segments = 20
        self.tx = np.linspace(-200, 800, n_timepoints)
        data = np.random.randn(n_timepoints, n_segments)
        
        self.seg = SegmentedEyeData.from_masked_array(
            data=data,
            tx=self.tx,
            variable="left_pupil",
            name="test_plot"
        )
    
    def test_plot_runs(self):
        """Test that plot method runs without error"""
        fig, ax = plt.subplots()
        self.seg.plot(ax=ax)
        plt.close(fig)
    
    def test_plot_standalone(self):
        """Test standalone plot sets title and ylabel"""
        fig, ax = plt.subplots()
        returned_ax = self.seg.plot()
        self.assertIsNotNone(returned_ax)
        self.assertEqual(returned_ax.get_title(), "test_plot")
        plt.close(fig)
    
    def test_plot_with_ax_no_title(self):
        """Test that title is not set when ax is provided"""
        fig, ax = plt.subplots()
        ax.set_title("Original Title")
        self.seg.plot(ax=ax)
        # Title should remain unchanged when ax is provided
        self.assertEqual(ax.get_title(), "Original Title")
        plt.close(fig)
    
    def test_plot_with_kwargs(self):
        """Test plot with matplotlib kwargs"""
        fig, ax = plt.subplots()
        self.seg.plot(ax=ax, color='red', linewidth=2, linestyle='--')
        plt.close(fig)
    
    def test_plot_with_custom_label(self):
        """Test plot with custom label for legend"""
        fig, ax = plt.subplots()
        self.seg.plot(ax=ax, label="Custom Label", show_legend=True)
        # Check that the label appears in legend handles
        handles, labels = ax.get_legend_handles_labels()
        self.assertIn("Custom Label", labels)
        plt.close(fig)
    
    def test_plot_overlay_multiple(self):
        """Test overlaying multiple segments"""
        # Create second segment
        seg2 = SegmentedEyeData.from_masked_array(
            data=np.random.randn(100, 20),
            tx=self.tx,
            variable="right_pupil",
            name="test_plot2"
        )
        
        fig, ax = plt.subplots()
        self.seg.plot(ax=ax, color='blue', label='Left')
        seg2.plot(ax=ax, color='red', label='Right')
        ax.legend()
        
        handles, labels = ax.get_legend_handles_labels()
        self.assertEqual(len(labels), 2)
        plt.close(fig)
    
    def test_plot_with_missing_overlay(self):
        """Test plot with missing data overlay in overlay mode"""
        fig, ax = plt.subplots()
        self.seg.plot(ax=ax, show_missing=True, show_legend=True)
        # Should have legend even with show_missing
        self.assertIsNotNone(ax.get_legend())
        plt.close(fig)
    
    def test_plot_standalone_shows_missing_by_default(self):
        """Test that standalone mode shows missing data by default"""
        fig = plt.figure()
        self.seg.plot()  # standalone mode
        # Should have two axes (main + twinx for missing)
        self.assertEqual(len(fig.axes), 2)
        plt.close(fig)
    
    def test_plot_returns_axes(self):
        """Test that plot returns the axes object"""
        fig, ax = plt.subplots()
        returned_ax = self.seg.plot(ax=ax)
        self.assertIs(returned_ax, ax)
        plt.close(fig)
    
    def test_plot_no_zero_line(self):
        """Test plot without zero line"""
        fig, ax = plt.subplots()
        self.seg.plot(ax=ax, show_zero_line=False)
        plt.close(fig)


class TestSegmentedEyeDataIO(unittest.TestCase):
    """Test file I/O functionality"""
    
    def test_write_and_read(self):
        """Test writing and reading SegmentedEyeData"""
        import tempfile
        import os
        
        n_timepoints = 50
        n_segments = 5
        tx = np.linspace(-100, 400, n_timepoints)
        data = np.random.randn(n_timepoints, n_segments)
        
        seg = SegmentedEyeData.from_masked_array(
            data=data,
            tx=tx,
            variable="left_pupil",
            name="io_test",
            sampling_rate=500.0
        )
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            fname = f.name
        
        try:
            seg.write_file(fname)
            
            # Read back
            seg_loaded = SegmentedEyeData.from_file(fname)
            
            self.assertEqual(seg_loaded.name, "io_test")
            self.assertEqual(seg_loaded.variable, "left_pupil")
            self.assertEqual(seg_loaded.n_timepoints, n_timepoints)
            self.assertEqual(seg_loaded.n_segments, n_segments)
            self.assertEqual(seg_loaded.sampling_rate, 500.0)
            np.testing.assert_array_almost_equal(seg_loaded.tx, tx)
            np.testing.assert_array_almost_equal(seg_loaded.data, seg.data)
        finally:
            os.unlink(fname)


class TestSegmentedEyeDataFromIntervals(unittest.TestCase):
    """Test integration with Intervals class"""
    
    def test_intervals_metadata_preserved(self):
        """Test that Intervals reference is stored"""
        intervals = Intervals(
            [(0, 100), (200, 300)],
            units="ms",
            label="test_intervals",
            event_labels=["ev1", "ev2"]
        )
        
        tx = np.linspace(0, 100, 50)
        data = np.random.randn(50, 2)
        
        seg = SegmentedEyeData.from_masked_array(
            data=data,
            tx=tx,
            variable="left_pupil",
            intervals=intervals
        )
        
        self.assertIs(seg.intervals, intervals)
        self.assertEqual(seg.intervals.label, "test_intervals")
    
    def test_name_from_intervals_label(self):
        """Test that name defaults to intervals label"""
        # Create PupilData
        data = pp.PupilData(
            sampling_rate=100,
            left_pupil=np.random.randn(500) + 500,
            event_onsets=[1000, 2000, 3000],
            event_labels=["cue"] * 3
        )
        
        intervals = data.get_intervals("cue", interval=(-100, 200), units="ms")
        segments = data.get_segments(intervals, "left_pupil")
        
        # Name should come from intervals label which was set by get_intervals
        self.assertEqual(segments.name, "cue")


class TestSegmentedEyeDataTimeAxis(unittest.TestCase):
    """Test that time axis is correctly computed with interval_window"""
    
    def test_time_axis_is_relative_to_event(self):
        """Test that time axis uses interval_window for event-locked intervals"""
        # Create PupilData with events
        n_samples = 1000
        fs = 100
        pupil = np.random.randn(n_samples) + 500
        event_onsets = [2000, 5000, 8000]  # in ms (2s, 5s, 8s)
        event_labels = ["stim"] * 3
        
        data = pp.PupilData(
            sampling_rate=fs,
            left_pupil=pupil,
            event_onsets=event_onsets,
            event_labels=event_labels
        )
        
        # Get intervals with window (-200, 800)
        intervals = data.get_intervals("stim", interval=(-200, 800), units="ms")
        
        # Verify interval_window is set
        self.assertEqual(intervals.interval_window, (-200, 800))
        
        # Get segments
        segments = data.get_segments(intervals, "left_pupil")
        
        # Time axis should go from -200 to 800, NOT from absolute times
        self.assertLess(segments.tx[0], 0)  # Should start negative
        self.assertGreater(segments.tx[-1], 0)  # Should end positive
        self.assertAlmostEqual(segments.tx[0], -200, delta=50)  # Close to -200
        self.assertAlmostEqual(segments.tx[-1], 800, delta=50)  # Close to 800
        
        # Zero should be approximately in the middle of the window
        zero_idx = np.argmin(np.abs(segments.tx))
        expected_zero_idx = int(segments.n_timepoints * 200 / 1000)  # 200ms into a 1000ms window
        self.assertAlmostEqual(zero_idx, expected_zero_idx, delta=5)
    
    def test_interval_window_preserved_across_units(self):
        """Test that interval_window is preserved when converting units"""
        intervals = Intervals(
            [(0, 100), (200, 300)],
            units="ms",
            label="test",
            sampling_rate=1000,
            interval_window=(-50, 50)
        )
        
        # Convert to seconds
        intervals_sec = intervals.to_units("sec")
        
        # interval_window should be converted too
        self.assertIsNotNone(intervals_sec.interval_window)
        self.assertAlmostEqual(intervals_sec.interval_window[0], -0.05)
        self.assertAlmostEqual(intervals_sec.interval_window[1], 0.05)
    
    def test_interval_window_updated_by_pad(self):
        """Test that padding updates interval_window"""
        intervals = Intervals(
            [(100, 200), (300, 400)],
            units="ms",
            label="test",
            interval_window=(-50, 50)
        )
        
        # Pad by 10ms on each side
        padded = intervals.pad(left=10, right=10)
        
        # interval_window should be expanded
        self.assertEqual(padded.interval_window, (-60, 60))


class TestGroupLevelSegmentedData(unittest.TestCase):
    """Test GroupLevelSegmentedData class"""
    
    def setUp(self):
        """Create test segment data for multiple subjects"""
        self.n_timepoints = 100
        self.n_segments = 20
        self.tx = np.linspace(-200, 800, self.n_timepoints)
        
        # Create 3 subjects with different data
        self.segments = []
        for i in range(3):
            data = np.random.randn(self.n_timepoints, self.n_segments) + i * 10
            # Add some missing data
            mask = np.zeros((self.n_timepoints, self.n_segments), dtype=bool)
            mask[0:5, :] = True  # First 5 timepoints masked
            
            seg = SegmentedEyeData.from_masked_array(
                data=data,
                tx=self.tx,
                variable="left_pupil",
                name=f"subject_{i+1}",
                mask=mask
            )
            self.segments.append(seg)
    
    def test_from_segments_aggregate(self):
        """Test combining segments with aggregation"""
        group = GroupLevelSegmentedData.from_segments(self.segments, meanfct=np.mean)
        
        self.assertIsInstance(group, GroupLevelSegmentedData)
        self.assertEqual(group.n_subjects, 3)
        self.assertEqual(group.n_timepoints, self.n_timepoints)
        # In aggregate mode, n_segments = n_subjects
        self.assertEqual(group.n_segments, 3)
        self.assertEqual(group.variable, "left_pupil")
    
    def test_from_segments_stack(self):
        """Test combining segments with stacking (meanfct=None)"""
        group = GroupLevelSegmentedData.from_segments(self.segments, meanfct=None)
        
        self.assertIsInstance(group, GroupLevelSegmentedData)
        self.assertEqual(group.n_subjects, 3)
        self.assertEqual(group.n_timepoints, self.n_timepoints)
        # In stack mode, n_segments = sum of all segments
        self.assertEqual(group.n_segments, self.n_segments * 3)
    
    def test_mask_percent_computed(self):
        """Test that mask_percent is correctly computed"""
        group = GroupLevelSegmentedData.from_segments(self.segments)
        
        self.assertEqual(group.mask_percent.shape, (self.n_timepoints,))
        # First 5 timepoints were masked for all subjects
        # 100% missing in those timepoints
        np.testing.assert_array_almost_equal(group.mask_percent[0:5], 100.0)
        # Rest should be 0%
        np.testing.assert_array_almost_equal(group.mask_percent[5:], 0.0)
    
    def test_data_not_masked_at_group_level(self):
        """Test that group-level data has no mask (info in mask_percent)"""
        group = GroupLevelSegmentedData.from_segments(self.segments)
        
        # Data should not be masked
        self.assertFalse(np.any(group.data.mask))
    
    def test_summary(self):
        """Test summary includes group-level info"""
        group = GroupLevelSegmentedData.from_segments(self.segments)
        summary = group.summary()
        
        self.assertIn('n_subjects', summary)
        self.assertIn('mean_mask_percent', summary)
        self.assertEqual(summary['n_subjects'], 3)
    
    def test_repr(self):
        """Test string representation"""
        group = GroupLevelSegmentedData.from_segments(self.segments)
        repr_str = repr(group)
        
        self.assertIn("GroupLevelSegmentedData", repr_str)
        self.assertIn("n_subjects", repr_str)
    
    def test_repr_html(self):
        """Test HTML representation"""
        group = GroupLevelSegmentedData.from_segments(self.segments)
        html = group._repr_html_()
        
        self.assertIn("GroupLevelSegmentedData", html)
        self.assertIn("Subjects", html)
    
    def test_plot_runs(self):
        """Test that plot method runs without error"""
        group = GroupLevelSegmentedData.from_segments(self.segments)
        fig, ax = plt.subplots()
        group.plot(ax=ax)
        plt.close(fig)
    
    def test_plot_standalone(self):
        """Test standalone plot"""
        group = GroupLevelSegmentedData.from_segments(self.segments)
        fig = plt.figure()
        ax = group.plot()
        self.assertIsNotNone(ax)
        plt.close(fig)
    
    def test_validation_variable_mismatch(self):
        """Test that mismatched variables raise error"""
        # Create segment with different variable
        bad_seg = SegmentedEyeData.from_masked_array(
            data=np.random.randn(self.n_timepoints, self.n_segments),
            tx=self.tx,
            variable="right_pupil",
            name="bad_subject"
        )
        
        with self.assertRaises(ValueError):
            GroupLevelSegmentedData.from_segments([self.segments[0], bad_seg])
    
    def test_validation_timepoints_mismatch(self):
        """Test that mismatched timepoints raise error"""
        # Create segment with different number of timepoints
        bad_seg = SegmentedEyeData.from_masked_array(
            data=np.random.randn(50, self.n_segments),  # Different n_timepoints
            tx=np.linspace(-200, 800, 50),
            variable="left_pupil",
            name="bad_subject"
        )
        
        with self.assertRaises(ValueError):
            GroupLevelSegmentedData.from_segments([self.segments[0], bad_seg])
    
    def test_validation_empty_list(self):
        """Test that empty list raises error"""
        with self.assertRaises(ValueError):
            GroupLevelSegmentedData.from_segments([])
    
    def test_custom_name(self):
        """Test custom name parameter"""
        group = GroupLevelSegmentedData.from_segments(
            self.segments, 
            name="my_group"
        )
        self.assertEqual(group.name, "my_group")
    
    def test_default_name(self):
        """Test default name generation"""
        group = GroupLevelSegmentedData.from_segments(self.segments)
        self.assertEqual(group.name, "group_subject_1")


class TestGroupSegmentsFunction(unittest.TestCase):
    """Test group_segments convenience function"""
    
    def setUp(self):
        """Create test segment data"""
        self.n_timepoints = 50
        self.n_segments = 10
        self.tx = np.linspace(-100, 400, self.n_timepoints)
        
        self.segments = []
        for i in range(3):
            seg = SegmentedEyeData.from_masked_array(
                data=np.random.randn(self.n_timepoints, self.n_segments),
                tx=self.tx,
                variable="left_pupil",
                name=f"subj_{i}"
            )
            self.segments.append(seg)
    
    def test_from_list(self):
        """Test group_segments with list input"""
        group = group_segments(self.segments)
        
        self.assertIsInstance(group, GroupLevelSegmentedData)
        self.assertEqual(group.n_subjects, 3)
    
    def test_from_dict(self):
        """Test group_segments with dict input"""
        seg_dict = {f"S{i:02d}": seg for i, seg in enumerate(self.segments)}
        group = group_segments(seg_dict)
        
        self.assertIsInstance(group, GroupLevelSegmentedData)
        self.assertEqual(group.n_subjects, 3)
    
    def test_stack_mode(self):
        """Test group_segments with meanfct=None (stack mode)"""
        group = group_segments(self.segments, meanfct=None)
        
        self.assertEqual(group.n_segments, self.n_segments * 3)
    
    def test_custom_name(self):
        """Test group_segments with custom name"""
        group = group_segments(self.segments, name="my_group")
        
        self.assertEqual(group.name, "my_group")
    
    def test_available_via_pp(self):
        """Test that group_segments is available via pp namespace"""
        self.assertTrue(hasattr(pp, 'group_segments'))
        self.assertTrue(callable(pp.group_segments))


if __name__ == '__main__':
    unittest.main()

