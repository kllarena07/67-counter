#!/usr/bin/env python3
"""
Simple test to verify the counting logic of HandCrossingCounter
"""

import time
import unittest
from unittest.mock import patch
from hand_crossing_counter import HandCrossingCounter


class TestHandCrossingCounter(unittest.TestCase):
    def setUp(self):
        self.counter = HandCrossingCounter()
    
    def test_initial_state(self):
        """Test that counter starts with correct initial values"""
        self.assertEqual(self.counter.crossing_count, 0)
        self.assertIsNone(self.counter.left_hand_y)
        self.assertIsNone(self.counter.right_hand_y)
    
    def test_counts_per_minute_empty(self):
        """Test counts per minute calculation with no data"""
        current_time = time.time()
        rate = self.counter.get_counts_per_minute(current_time)
        self.assertEqual(rate, 0.0)
    
    def test_counts_per_minute_with_data(self):
        """Test counts per minute calculation with some data"""
        current_time = time.time()
        
        # Set start time to 30 seconds ago
        self.counter.start_time = current_time - 30
        
        # Add some timestamps 
        self.counter.count_timestamps = [
            current_time - 25,
            current_time - 20,
            current_time - 15,
            current_time - 10,
            current_time - 5
        ]
        
        # Should extrapolate to 60 seconds: 5 counts in 30 seconds = 10 per minute
        rate = self.counter.get_counts_per_minute(current_time)
        self.assertAlmostEqual(rate, 10.0, places=1)
    
    def test_counts_per_minute_full_minute(self):
        """Test counts per minute with full 60 seconds of data"""
        current_time = time.time()
        self.counter.start_time = current_time - 65  # More than 60 seconds ago
        
        # Add timestamps within last 60 seconds
        self.counter.count_timestamps = [
            current_time - 50,
            current_time - 40,
            current_time - 30,
            current_time - 20,
            current_time - 10
        ]
        
        # Should return exact count within last 60 seconds
        rate = self.counter.get_counts_per_minute(current_time)
        self.assertEqual(rate, 5)
    
    def test_reset_counter(self):
        """Test that reset clears all counters"""
        # Set some state
        self.counter.crossing_count = 10
        self.counter.count_timestamps = [time.time()]
        
        # Reset
        self.counter.reset_counter()
        
        # Verify reset
        self.assertEqual(self.counter.crossing_count, 0)
        self.assertEqual(len(self.counter.count_timestamps), 0)
    
    @patch('builtins.print')  # Mock print to avoid output during tests
    def test_detect_crossing_basic(self, mock_print):
        """Test basic crossing detection logic"""
        current_time = time.time()
        
        # Set initial positions (no previous positions)
        self.counter.left_hand_y = 0.3
        self.counter.right_hand_y = 0.7
        
        # Should detect that left hand is above right hand
        result = self.counter.detect_crossing(current_time)
        self.assertTrue(result)
        self.assertEqual(self.counter.crossing_count, 1)
    
    @patch('builtins.print')
    def test_detect_crossing_movement(self, mock_print):
        """Test crossing detection with hand movement"""
        current_time = time.time()
        
        # Set previous positions: left was below right
        self.counter.previous_left_y = 0.7
        self.counter.previous_right_y = 0.3
        
        # Set current positions: left is now above right (crossing occurred)
        self.counter.left_hand_y = 0.3
        self.counter.right_hand_y = 0.7
        
        # Should detect crossing
        result = self.counter.detect_crossing(current_time)
        self.assertTrue(result)
        self.assertEqual(self.counter.crossing_count, 1)


if __name__ == '__main__':
    unittest.main()