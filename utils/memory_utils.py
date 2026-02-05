#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Memory Management Utilities

Fix memory fragmentation issues in neighborhood.py without changing the patent algorithm.
"""

import gc
import sys
import ctypes

def force_memory_cleanup():
    """
    Force aggressive memory cleanup.

    This helps reduce fragmentation by:
    1. Running Python garbage collector
    2. On Windows, calling HeapCompact to defragment the heap
    """
    # Python GC
    gc.collect()
    gc.collect()
    gc.collect()

    # Windows-specific: try to compact heap
    if sys.platform == 'win32':
        try:
            # Get process heap and compact it
            kernel32 = ctypes.windll.kernel32
            heap = kernel32.GetProcessHeap()
            kernel32.HeapCompact(heap, 0)
        except Exception:
            pass  # Silently fail if not available


def setup_memory_management():
    """
    Configure memory management settings for long training runs.

    Call this once at the start of training.
    """
    # Disable GC during critical sections, but enable for explicit calls
    gc.enable()

    # Set GC thresholds to be more aggressive
    # Default is (700, 10, 10), we make it more frequent
    gc.set_threshold(500, 5, 5)

    print("[Memory] Aggressive GC enabled (threshold: 500, 5, 5)")


class MemoryManager:
    """
    Memory manager that periodically cleans up memory during training.

    Usage:
        mm = MemoryManager(cleanup_interval=50)
        for step in range(total_steps):
            # ... training step ...
            mm.step()  # Will cleanup every 50 steps
    """

    def __init__(self, cleanup_interval=50, verbose=True):
        """
        Args:
            cleanup_interval: Number of steps between cleanups
            verbose: Whether to print cleanup messages
        """
        self.cleanup_interval = cleanup_interval
        self.verbose = verbose
        self.step_count = 0
        self.cleanup_count = 0

    def step(self):
        """Call this after each training step."""
        self.step_count += 1

        if self.step_count % self.cleanup_interval == 0:
            self._cleanup()

    def _cleanup(self):
        """Perform memory cleanup."""
        force_memory_cleanup()
        self.cleanup_count += 1

        if self.verbose:
            try:
                import psutil
                import os
                mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                print(f"[Memory] Cleanup #{self.cleanup_count} at step {self.step_count}, RSS={mem_mb:.1f}MB")
            except ImportError:
                print(f"[Memory] Cleanup #{self.cleanup_count} at step {self.step_count}")


# Pre-allocated memory pools for neighborhood calculations
class NeighborhoodMemoryPool:
    """
    Pre-allocated memory pool for neighborhood calculations.

    This reduces fragmentation by reusing arrays instead of constantly
    allocating and deallocating.
    """

    def __init__(self, batch_size, time_steps, n_features, latent_dim, k):
        """
        Pre-allocate arrays for a fixed batch/time/feature size.

        Args:
            batch_size: Maximum batch size
            time_steps: Number of time steps (e.g., 48)
            n_features: Number of features (e.g., 44)
            latent_dim: Latent dimension (e.g., 64)
            k: Number of neighbors
        """
        import numpy as np

        self.batch_size = batch_size
        self.time_steps = time_steps
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.k = k

        # Pre-allocate spatial neighborhood arrays (per sample)
        # These are the O(T^2 * N) arrays that cause fragmentation
        self.common_mask = np.zeros((time_steps, time_steps, n_features), dtype=np.float32)
        self.diff = np.zeros((time_steps, time_steps, n_features), dtype=np.float32)
        self.squared_diff = np.zeros((time_steps, time_steps, n_features), dtype=np.float32)
        self.distances = np.zeros((time_steps, time_steps), dtype=np.float32)
        self.n_common = np.zeros((time_steps, time_steps), dtype=np.float32)

        # Pre-allocate temporal neighborhood arrays (per sample)
        self.var_distances = np.zeros((n_features, n_features), dtype=np.float32)
        self.z_var = np.zeros((n_features, latent_dim), dtype=np.float32)
        self.z_var_agg = np.zeros((n_features, latent_dim), dtype=np.float32)
        self.z_var_neighbors = np.zeros((n_features, k, latent_dim), dtype=np.float32)
        self.z_time = np.zeros((time_steps, latent_dim), dtype=np.float32)

        # Results arrays (per batch)
        self.indices_list = []
        self.weights_list = []

        print(f"[MemoryPool] Pre-allocated arrays for T={time_steps}, N={n_features}, latent={latent_dim}")
        total_mb = (
            self.common_mask.nbytes +
            self.diff.nbytes +
            self.squared_diff.nbytes +
            self.distances.nbytes +
            self.n_common.nbytes +
            self.var_distances.nbytes +
            self.z_var.nbytes +
            self.z_var_agg.nbytes +
            self.z_var_neighbors.nbytes +
            self.z_time.nbytes
        ) / 1024 / 1024
        print(f"[MemoryPool] Total pre-allocated: {total_mb:.2f} MB")

    def reset_batch(self):
        """Reset batch-level accumulators."""
        self.indices_list = []
        self.weights_list = []

    def get_spatial_arrays(self):
        """Get pre-allocated arrays for spatial neighborhood calculation."""
        # Zero out arrays for reuse
        self.common_mask.fill(0)
        self.diff.fill(0)
        self.squared_diff.fill(0)
        self.distances.fill(0)
        self.n_common.fill(0)

        return (self.common_mask, self.diff, self.squared_diff,
                self.distances, self.n_common)

    def get_temporal_arrays(self):
        """Get pre-allocated arrays for temporal neighborhood calculation."""
        # Zero out arrays for reuse
        self.var_distances.fill(0)
        self.z_var.fill(0)
        self.z_var_agg.fill(0)
        self.z_var_neighbors.fill(0)
        self.z_time.fill(0)

        return (self.var_distances, self.z_var, self.z_var_agg,
                self.z_var_neighbors, self.z_time)


if __name__ == '__main__':
    # Test memory management
    print("Testing memory management utilities...")

    setup_memory_management()

    mm = MemoryManager(cleanup_interval=10)

    import numpy as np

    # Simulate training loop
    for step in range(50):
        # Simulate creating temporary arrays (like neighborhood.py does)
        temp1 = np.random.randn(48, 48, 44).astype(np.float32)
        temp2 = np.random.randn(48, 48, 44).astype(np.float32)
        temp3 = temp1 * temp2

        mm.step()

    print("\nTest complete!")
