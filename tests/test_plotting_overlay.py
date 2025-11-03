"""
Tests for overlay plotting functionality.
These tests verify that label_prefix and style parameters work correctly for overlaying data.
"""
import unittest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "..")
import pypillometry as pp
import numpy as np


class TestPlottingOverlay(unittest.TestCase):
    """Test overlay plotting with label_prefix and style parameters"""
    
    def setUp(self):
        """Create test data"""
        self.data = pp.get_example_data("rlmw_002_short")
        # Create a modified version to simulate processed data
        self.data2 = self.data.copy()
        plt.close('all')
    
    def tearDown(self):
        """Clean up after each test"""
        plt.close('all')
    
    def test_plot_timeseries_with_label_prefix(self):
        """Test plot_timeseries with label_prefix"""
        fig = plt.figure()
        self.data.plot.plot_timeseries(
            variables=['pupil'],
            label_prefix="original"
        )
        
        # Check that lines were created with correct labels
        ax = fig.axes[0]
        labels = [line.get_label() for line in ax.get_lines()]
        self.assertTrue(any('original' in label for label in labels))
    
    def test_plot_timeseries_overlay_two_datasets(self):
        """Test overlaying two datasets with different prefixes"""
        fig = plt.figure()
        
        # Plot original data
        self.data.plot.plot_timeseries(
            variables=['pupil'],
            label_prefix="original"
        )
        
        # Plot modified data on same axes
        self.data2.plot.plot_timeseries(
            variables=['pupil'],
            label_prefix="processed"
        )
        
        # Check that both datasets are plotted
        ax = fig.axes[0]
        labels = [line.get_label() for line in ax.get_lines()]
        self.assertTrue(any('original' in label for label in labels))
        self.assertTrue(any('processed' in label for label in labels))
    
    def test_plot_timeseries_with_style(self):
        """Test plot_timeseries with custom style"""
        fig = plt.figure()
        custom_style = {'color': 'red', 'linestyle': '--', 'alpha': 0.7}
        
        self.data.plot.plot_timeseries(
            variables=['pupil'],
            style=custom_style
        )
        
        # Check that style was applied
        ax = fig.axes[0]
        line = ax.get_lines()[0]
        self.assertEqual(line.get_color(), 'red')
        self.assertEqual(line.get_linestyle(), '--')
        self.assertAlmostEqual(line.get_alpha(), 0.7)
    
    def test_plot_timeseries_overlay_with_styles(self):
        """Test overlaying with different styles"""
        fig = plt.figure()
        
        # Plot with solid line (only left eye)
        self.data.plot.plot_timeseries(
            variables=['pupil'],
            eyes=['left'],
            label_prefix="solid",
            style={'linestyle': '-', 'color': 'blue'}
        )
        
        # Plot with dashed line (only left eye)
        self.data2.plot.plot_timeseries(
            variables=['pupil'],
            eyes=['left'],
            label_prefix="dashed",
            style={'linestyle': '--', 'color': 'red'}
        )
        
        ax = fig.axes[0]
        lines = ax.get_lines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0].get_linestyle(), '-')
        self.assertEqual(lines[1].get_linestyle(), '--')
    
    def test_pupil_plot_with_label_prefix(self):
        """Test pupil_plot with label_prefix"""
        fig = plt.figure()
        self.data.plot.pupil_plot(
            label_prefix="v1"
        )
        
        # Check that labels have prefix
        labels = [line.get_label() for line in plt.gca().get_lines()]
        self.assertTrue(any('v1' in label for label in labels))
    
    def test_pupil_plot_overlay_two_datasets(self):
        """Test overlaying two datasets with pupil_plot"""
        fig = plt.figure()
        
        self.data.plot.pupil_plot(
            label_prefix="original",
            style={'alpha': 0.8}
        )
        
        self.data2.plot.pupil_plot(
            label_prefix="interpolated",
            style={'alpha': 0.6, 'linestyle': '--'}
        )
        
        labels = [line.get_label() for line in plt.gca().get_lines()]
        self.assertTrue(any('original' in label for label in labels))
        self.assertTrue(any('interpolated' in label for label in labels))
    
    def test_pupil_plot_with_style(self):
        """Test pupil_plot with custom style"""
        fig = plt.figure()
        self.data.plot.pupil_plot(
            style={'color': 'green', 'linewidth': 2}
        )
        
        line = plt.gca().get_lines()[0]
        self.assertEqual(line.get_color(), 'green')
        self.assertEqual(line.get_linewidth(), 2)
    
    def test_overlay_without_prefix(self):
        """Test that overlay works without prefix (default behavior)"""
        fig = plt.figure()
        
        # Without prefix
        self.data.plot.plot_timeseries(variables=['pupil'])
        
        # Should still work and create labels
        ax = fig.axes[0]
        labels = [line.get_label() for line in ax.get_lines()]
        self.assertGreater(len(labels), 0)
        # Labels should not have underscores at start (no empty prefix)
        for label in labels:
            self.assertFalse(label.startswith('_'))


class TestPlottingOverlayMultipleEyes(unittest.TestCase):
    """Test overlay with multiple eyes"""
    
    def setUp(self):
        """Create test data with both eyes"""
        self.data = pp.PupilData(
            sampling_rate=100,
            left_pupil=np.sin(np.linspace(0, 10*np.pi, 1000)),
            right_pupil=np.cos(np.linspace(0, 10*np.pi, 1000))
        )
        self.data2 = self.data.copy()
        plt.close('all')
    
    def tearDown(self):
        """Clean up after each test"""
        plt.close('all')
    
    def test_overlay_both_eyes(self):
        """Test overlaying with both eyes"""
        fig = plt.figure()
        
        self.data.plot.pupil_plot(
            eyes=['left', 'right'],
            label_prefix="v1"
        )
        
        self.data2.plot.pupil_plot(
            eyes=['left', 'right'],
            label_prefix="v2"
        )
        
        labels = [line.get_label() for line in plt.gca().get_lines()]
        # Should have 4 lines: v1_left, v1_right, v2_left, v2_right
        self.assertEqual(len(labels), 4)
        self.assertTrue('v1_left' in labels)
        self.assertTrue('v1_right' in labels)
        self.assertTrue('v2_left' in labels)
        self.assertTrue('v2_right' in labels)


class TestPlottingOverlayMultipleVariables(unittest.TestCase):
    """Test overlay with multiple variables"""
    
    def setUp(self):
        """Create test data"""
        self.data = pp.get_example_data("rlmw_002_short")
        self.data2 = self.data.copy()
        plt.close('all')
    
    def tearDown(self):
        """Clean up after each test"""
        plt.close('all')
    
    def test_overlay_multiple_variables(self):
        """Test overlaying with multiple variables"""
        fig = plt.figure()
        
        # Get available variables
        available_vars = list(set([k.split('_')[1] for k in self.data.data.keys() if '_' in k]))
        if len(available_vars) >= 2:
            vars_to_plot = available_vars[:2]
            
            self.data.plot.plot_timeseries(
                variables=vars_to_plot,
                label_prefix="original"
            )
            
            self.data2.plot.plot_timeseries(
                variables=vars_to_plot,
                label_prefix="modified"
            )
            
            # Check each subplot has overlaid data
            for ax in fig.axes:
                labels = [line.get_label() for line in ax.get_lines()]
                self.assertTrue(any('original' in label for label in labels))
                self.assertTrue(any('modified' in label for label in labels))


if __name__ == '__main__':
    unittest.main()

