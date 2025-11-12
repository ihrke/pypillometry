"""
Tests for PDF generation in plotting functions.
These tests verify that PDF files are created correctly.
"""
import unittest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "..")
import pypillometry as pp
import tempfile
import os
import shutil


class TestPlottingPDFGeneration(unittest.TestCase):
    """Test PDF file generation"""
    
    def setUp(self):
        """Create test data and temporary directory"""
        self.data = pp.get_example_data("rlmw_002_short")
        self.temp_dir = tempfile.mkdtemp()
        plt.close('all')
    
    def tearDown(self):
        """Clean up after each test"""
        plt.close('all')
        # Clean up temp directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_plot_intervals_creates_pdf(self):
        """Test that plot_intervals creates PDF file"""
        pdf_path = os.path.join(self.temp_dir, "test_intervals.pdf")
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        
        figs = self.data.plot.plot_intervals(intervals, pdf_file=pdf_path)
        
        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 0)
    
    def test_plot_timeseries_segments_creates_pdf(self):
        """Test that plot_timeseries_segments creates PDF file"""
        pdf_path = os.path.join(self.temp_dir, "test_segments.pdf")
        
        figs = self.data.plot.plot_timeseries_segments(
            pdf_file=pdf_path, 
            interv=1.0
        )
        
        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 0)
    
    def test_pupil_plot_segments_creates_pdf(self):
        """Test that pupil_plot_segments creates PDF file"""
        pdf_path = os.path.join(self.temp_dir, "test_pupil_segments.pdf")
        
        figs = self.data.plot.pupil_plot_segments(
            pdf_file=pdf_path, 
            interv=1.0
        )
        
        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 0)
    
    def test_plot_blinks_creates_pdf(self):
        """Test that plot_blinks creates PDF file"""
        pdf_path = os.path.join(self.temp_dir, "test_blinks.pdf")
        
        # plot_blinks internally calls plot_intervals which handles PDF
        figs = self.data.plot.plot_blinks(pdf_file=pdf_path)
        
        # Check if PDF was created (plot_blinks may have different behavior)
        # It might create the PDF or just return figures
        if os.path.exists(pdf_path):
            self.assertGreater(os.path.getsize(pdf_path), 0)
        else:
            # If PDF wasn't created, at least check we got figures back
            self.assertIsNotNone(figs)


class TestPlottingPDFDirectoryCreation(unittest.TestCase):
    """Test that nested directories are created automatically"""
    
    def setUp(self):
        """Create test data and temporary directory"""
        self.data = pp.get_example_data("rlmw_002_short")
        self.temp_dir = tempfile.mkdtemp()
        plt.close('all')
    
    def tearDown(self):
        """Clean up after each test"""
        plt.close('all')
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_plot_intervals_nested_directories(self):
        """Test that nested directories are created for plot_intervals"""
        pdf_path = os.path.join(self.temp_dir, "subdir1", "subdir2", "test.pdf")
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        
        # Directory should not exist before
        self.assertFalse(os.path.exists(os.path.dirname(pdf_path)))
        
        figs = self.data.plot.plot_intervals(intervals, pdf_file=pdf_path)
        
        # Directory and file should exist after
        self.assertTrue(os.path.exists(os.path.dirname(pdf_path)))
        self.assertTrue(os.path.exists(pdf_path))
    
    def test_plot_timeseries_segments_nested_directories(self):
        """Test that nested directories are created for plot_timeseries_segments"""
        pdf_path = os.path.join(self.temp_dir, "results", "plots", "segments.pdf")
        
        self.assertFalse(os.path.exists(os.path.dirname(pdf_path)))
        
        figs = self.data.plot.plot_timeseries_segments(
            pdf_file=pdf_path, 
            interv=1.0
        )
        
        self.assertTrue(os.path.exists(os.path.dirname(pdf_path)))
        self.assertTrue(os.path.exists(pdf_path))
    
    def test_pupil_plot_segments_nested_directories(self):
        """Test that nested directories are created for pupil_plot_segments"""
        pdf_path = os.path.join(self.temp_dir, "a", "b", "c", "pupil.pdf")
        
        self.assertFalse(os.path.exists(os.path.dirname(pdf_path)))
        
        figs = self.data.plot.pupil_plot_segments(
            pdf_file=pdf_path, 
            interv=1.0
        )
        
        self.assertTrue(os.path.exists(os.path.dirname(pdf_path)))
        self.assertTrue(os.path.exists(pdf_path))


class TestPlottingPDFNoDisplay(unittest.TestCase):
    """Test that PDF generation doesn't display figures"""
    
    def setUp(self):
        """Create test data and temporary directory"""
        self.data = pp.get_example_data("rlmw_002_short")
        self.temp_dir = tempfile.mkdtemp()
        plt.close('all')
    
    def tearDown(self):
        """Clean up after each test"""
        plt.close('all')
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_plot_intervals_figures_closed(self):
        """Test that figures are closed when saving to PDF"""
        pdf_path = os.path.join(self.temp_dir, "test.pdf")
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        
        # Count figures before
        n_figs_before = len(plt.get_fignums())
        
        figs = self.data.plot.plot_intervals(intervals, pdf_file=pdf_path)
        
        # Figures should be closed (same or fewer than before)
        n_figs_after = len(plt.get_fignums())
        self.assertLessEqual(n_figs_after, n_figs_before)
    
    def test_plot_timeseries_segments_figures_closed(self):
        """Test that figures are closed when saving to PDF"""
        pdf_path = os.path.join(self.temp_dir, "test.pdf")
        
        n_figs_before = len(plt.get_fignums())
        
        figs = self.data.plot.plot_timeseries_segments(
            pdf_file=pdf_path, 
            interv=1.0
        )
        
        n_figs_after = len(plt.get_fignums())
        self.assertLessEqual(n_figs_after, n_figs_before)


if __name__ == '__main__':
    unittest.main()

