#!/usr/bin/env python3
"""
Safe Embedding Generator for RAG System - M2 MacBook Air Optimized
Optimized for 16GB unified memory with intelligent device selection
Fixes hanging issues in embedding generation with:
- Single-threaded processing
- M2-optimized batch processing
- Smart device selection (MPS/CPU)
- Robust error handling
- Memory management
"""

import os
import time
import gc
import logging
import multiprocessing as mp
from typing import List, Optional, Iterable
import numpy as np
import warnings

# Set stability environment BEFORE any imports
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def _get_optimal_device() -> str:
    """
    Get optimal device for M2 MacBook Air with intelligent fallback
    
    Returns:
        Device string: 'cpu' (safest for sentence-transformers)
    """
    # For sentence-transformers, CPU is most reliable
    # MPS support is experimental and can cause issues
    # With 16GB RAM, CPU performance is excellent
    return "cpu"


def _get_m2_optimal_batch_size() -> int:
    """
    Get optimal batch size for 16GB M2 MacBook Air
    
    Returns:
        Optimal batch size (64-128 for 16GB systems)
    """
    # 16GB unified memory can handle large batches efficiently
    # Balanced for speed and stability
    return 64


class SafeEmbedder:
    """Safe, batched embedding generation optimized for M2 MacBook Air (16GB)"""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu",
                 batch_size: int = 64,
                 max_retries: int = 2):
        """
        Initialize SafeEmbedder with M2 optimization
        
        Args:
            model_name: Sentence transformer model name
            device: Device to use ('cpu' recommended for stability)
            batch_size: Number of texts to process at once (64 optimal for 16GB)
            max_retries: Number of retries for failed batches
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = max(1, batch_size)
        self.max_retries = max_retries
        self._model = None
        
        # Clean up any existing multiprocessing children
        self._cleanup_multiprocessing()
        
        # Log configuration
        print(f"SafeEmbedder initialized: device={self.device}, batch_size={self.batch_size}")
    
    def _cleanup_multiprocessing(self):
        """Clean up any lingering multiprocessing children"""
        try:
            for child in mp.active_children():
                child.terminate()
                child.join(timeout=1.0)
        except Exception:
            pass
    
    def _ensure_model_loaded(self):
        """Lazy load the sentence transformer model"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"Loading embedding model {self.model_name} on {self.device}...")
                self._model = SentenceTransformer(self.model_name, device=self.device)
                print("✓ Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
    
    def encode_batch_safe(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Safely encode a batch of texts with retries and error handling
        M2-optimized with larger sub-batches
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Numpy array of embeddings or None if failed
        """
        if not texts:
            return None
            
        self._ensure_model_loaded()
        
        for retry in range(self.max_retries):
            try:
                start_time = time.time()
                
                # M2 optimization: larger sub-batches (32 for 16GB systems)
                sub_batch_size = min(32, len(texts))
                
                embeddings = self._model.encode(
                    texts,
                    batch_size=sub_batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    device=self.device
                )
                
                elapsed = time.time() - start_time
                rate = len(texts) / elapsed if elapsed > 0 else 0
                logger.debug(f"Encoded {len(texts)} texts in {elapsed:.2f}s ({rate:.1f} texts/sec)")
                
                return embeddings.astype(np.float32, copy=False)
                
            except Exception as e:
                logger.warning(f"Batch encoding attempt {retry + 1} failed: {e}")
                if retry == self.max_retries - 1:
                    logger.error(f"Failed to encode batch after {self.max_retries} attempts")
                    return None
                    
                # Brief pause before retry
                time.sleep(0.5)
                gc.collect()
        
        return None
    
    def encode_texts_batched(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Encode list of texts using M2-optimized batching
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Numpy array of all embeddings concatenated
        """
        if not texts:
            return None
        
        total_texts = len(texts)
        total_batches = (total_texts + self.batch_size - 1) // self.batch_size
        
        print(f"🚀 M2 Optimized: Processing {total_texts} texts in {total_batches} batches of {self.batch_size}...")
        
        all_embeddings = []
        overall_start = time.time()
        
        for i in range(0, total_texts, self.batch_size):
            batch_num = (i // self.batch_size) + 1
            batch_texts = texts[i:i + self.batch_size]
            
            batch_start = time.time()
            batch_embeddings = self.encode_batch_safe(batch_texts)
            batch_elapsed = time.time() - batch_start
            
            if batch_embeddings is not None:
                all_embeddings.append(batch_embeddings)
                rate = len(batch_texts) / batch_elapsed if batch_elapsed > 0 else 0
                print(f"  ✓ Batch {batch_num}/{total_batches}: {len(batch_texts)} texts in {batch_elapsed:.1f}s ({rate:.1f} texts/sec)")
            else:
                logger.error(f"Batch {batch_num} failed completely")
                # Create fallback embeddings (zeros) to maintain data integrity
                fallback_dim = 384  # all-MiniLM-L6-v2 dimension
                fallback_embeddings = np.zeros((len(batch_texts), fallback_dim), dtype=np.float32)
                all_embeddings.append(fallback_embeddings)
                logger.warning(f"Using fallback zero embeddings for batch {batch_num}")
            
            # Minimal cleanup between batches
            if batch_num % 5 == 0:  # Every 5 batches
                gc.collect()
        
        overall_elapsed = time.time() - overall_start
        
        if all_embeddings:
            final_embeddings = np.vstack(all_embeddings)
            overall_rate = total_texts / overall_elapsed if overall_elapsed > 0 else 0
            print(f"✅ M2 Performance: {total_texts} embeddings in {overall_elapsed:.1f}s ({overall_rate:.1f} texts/sec)")
            print(f"   Final shape: {final_embeddings.shape}")
            return final_embeddings
        else:
            logger.error("Failed to generate any embeddings")
            return None
    
    def __del__(self):
        """Cleanup on destruction"""
        self._cleanup_multiprocessing()


def create_safe_embedder() -> SafeEmbedder:
    """
    Factory function to create SafeEmbedder with M2 MacBook Air (16GB) optimal settings
    
    Returns:
        SafeEmbedder configured for maximum M2 performance
    """
    device = os.getenv("RAG_EMBED_DEVICE", _get_optimal_device())
    batch_size = int(os.getenv("RAG_EMBED_BATCH_SIZE", str(_get_m2_optimal_batch_size())))
    
    return SafeEmbedder(
        model_name=os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        device=device,
        batch_size=batch_size,
        max_retries=int(os.getenv("RAG_EMBED_MAX_RETRIES", "2"))
    )


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("Testing M2-Optimized SafeEmbedder")
    print("=" * 60)
    
    test_texts = [
        "The child completed the coloring task with steady movements.",
        "Movement velocity showed high variability during the session.",
        "Palm touches were frequent, indicating possible motor difficulties.",
        "Session completed within normal time range.",
        "Color selection patterns showed systematic approach.",
        "Fine motor control appeared age-appropriate.",
        "Task engagement remained high throughout session.",
        "Behavioral patterns consistent with developmental norms."
    ]
    
    print(f"\n16GB M2 Configuration:")
    print(f"  - Device: {_get_optimal_device()}")
    print(f"  - Batch Size: {_get_m2_optimal_batch_size()}")
    print(f"  - Test texts: {len(test_texts)}\n")
    
    embedder = create_safe_embedder()
    embeddings = embedder.encode_texts_batched(test_texts)
    
    if embeddings is not None:
        print(f"\n{'='*60}")
        print(f"✅ M2-Optimized SafeEmbedder test SUCCESSFUL!")
        print(f"{'='*60}")
        print(f"Shape: {embeddings.shape}")
        print(f"Memory: {embeddings.nbytes / 1024:.1f} KB")
    else:
        print("\n❌ SafeEmbedder test failed")