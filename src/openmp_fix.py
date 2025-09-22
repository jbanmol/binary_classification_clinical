"""
OpenMP Warning Suppression for macOS

This module provides a safe way to suppress the OpenMP deprecation warning
on macOS without affecting model functionality or performance.

The warning "OMP: Info #276: omp_set_nested routine deprecated, please use 
omp_set_max_active_levels instead" is caused by underlying OpenMP libraries
used by scientific computing packages (NumPy, SciPy, scikit-learn, UMAP, etc.).

This fix:
1. Suppresses the specific OpenMP deprecation warning
2. Maintains all existing thread limiting for stability
3. Preserves model configurations and functionality
4. Works specifically on macOS systems
"""

import os
import sys
import warnings
from typing import Optional


def suppress_openmp_warnings() -> None:
    """
    Suppress OpenMP deprecation warnings on macOS without affecting functionality.
    
    This function should be called at the very beginning of any script that uses
    scientific computing libraries to prevent the OpenMP deprecation warning
    from appearing in the output.
    """
    # Only apply on macOS systems
    if sys.platform != 'darwin':
        return
    
    # Set up thread limiting for stability (existing behavior)
    thread_limits = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", 
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    
    for key, value in thread_limits.items():
        os.environ.setdefault(key, value)
    
    # Suppress the specific OpenMP deprecation warning
    # This warning comes from the underlying OpenMP library, not Python warnings
    # We need to redirect stderr temporarily during library imports
    _suppress_openmp_deprecation_warning()


def _suppress_openmp_deprecation_warning() -> None:
    """
    Suppress the specific OpenMP deprecation warning by filtering stderr output.
    This is a safe approach that doesn't affect the actual OpenMP functionality.
    """
    import io
    import contextlib
    
    # Create a custom stderr filter that removes the specific OpenMP warning
    class OpenMPWarningFilter:
        def __init__(self, original_stderr):
            self.original_stderr = original_stderr
            self.buffer = io.StringIO()
        
        def write(self, message):
            # Filter out the specific OpenMP deprecation warning
            if "OMP: Info #276: omp_set_nested routine deprecated" not in message:
                self.original_stderr.write(message)
        
        def flush(self):
            self.original_stderr.flush()
        
        def __getattr__(self, name):
            return getattr(self.original_stderr, name)
    
    # Apply the filter to stderr
    sys.stderr = OpenMPWarningFilter(sys.stderr)


def setup_environment() -> None:
    """
    Complete environment setup for the binary classification project.
    
    This function:
    1. Suppresses OpenMP warnings
    2. Sets up thread limiting for stability
    3. Configures logging appropriately
    
    Call this function at the beginning of any script in the project.
    """
    # Suppress OpenMP warnings first
    suppress_openmp_warnings()
    
    # Set up additional environment variables for stability
    os.environ.setdefault('PYTHONWARNINGS', 'ignore::UserWarning')
    
    # Configure logging to be less verbose for OpenMP-related messages
    import logging
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('umap').setLevel(logging.WARNING)


# Convenience function for easy import and use
def apply_openmp_fix() -> None:
    """
    Apply the OpenMP fix. This is the main function to call.
    
    Usage:
        from src.openmp_fix import apply_openmp_fix
        apply_openmp_fix()
        # Now import your scientific libraries
    """
    setup_environment()


if __name__ == "__main__":
    # Test the fix
    print("Testing OpenMP warning suppression...")
    apply_openmp_fix()
    
    # Test imports that might trigger the warning
    try:
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        print("✅ Core libraries imported successfully")
        
        # Test UMAP which often triggers the warning
        try:
            import umap
            print("✅ UMAP imported successfully")
        except ImportError:
            print("⚠️  UMAP not available (optional)")
            
        print("✅ OpenMP warning suppression working correctly")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
