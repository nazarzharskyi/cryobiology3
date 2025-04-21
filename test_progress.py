"""
Test script to verify the progress bar functionality.

This script simulates a processing task with progress updates
to ensure the progress bar updates correctly in a single line.
"""

import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cellsegkit.pipeline.pipeline import _print_progress

def simulate_processing(total_steps=20):
    """Simulate a processing task with progress updates."""
    print("Starting simulation...")
    
    for i in range(total_steps + 1):
        # Calculate progress percentage
        progress_pct = (i / total_steps) * 100
        
        # Update progress bar
        if i < total_steps:
            message = f"Processing step {i+1}/{total_steps}"
        else:
            message = "Completed all steps"
            
        _print_progress(progress_pct, message)
        
        # Simulate processing time
        time.sleep(0.3)
    
    # Print final message on a new line
    print("\n\nSimulation completed successfully!")

if __name__ == "__main__":
    simulate_processing()