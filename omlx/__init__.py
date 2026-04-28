# SPDX-License-Identifier: Apache-2.0
"""
omlx: LLM inference server, optimized for your Mac

This package provides native Apple Silicon GPU acceleration using
Apple's MLX framework and mlx-lm for LLMs.

Features:
- Continuous batching via vLLM-style scheduler
- OpenAI-compatible API server
- Paged KV cache with prefix sharing
- Tiered cache (GPU + paged SSD offloading)
"""

from omlx._version import __version__

# Continuous batching engine (core functionality, no torch required)
from omlx.request import Request, RequestOutput, RequestStatus, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig, SchedulerOutput
from omlx.engine_core import EngineCore, AsyncEngineCore, EngineConfig
from omlx.cache.prefix_cache import BlockAwarePrefixCache
from omlx.cache.paged_cache import PagedCacheManager, CacheBlock, BlockTable
from omlx.cache.stats import PrefixCacheStats, PagedCacheStats
from omlx.model_registry import get_registry, ModelOwnershipError

# Metal Guard initialization (optional — silently skipped if not installed)
# Patches MLX at C level to prevent kernel panics on known-failing models.
# Must be imported before any MLX operations.
try:
    import metal_guard
    # Apply C-level defensive patches (prepare_count_underflow, etc.)
    metal_guard.install_upstream_defensive_patches()
    # Log system audit at startup for debugging
    metal_guard.log_system_audit_at_startup()
    _METAL_GUARD_AVAILABLE = True
except ImportError:
    _METAL_GUARD_AVAILABLE = False

# Backward compatibility alias
CacheStats = PagedCacheStats

__all__ = [
    # Request management
    "Request",
    "RequestOutput",
    "RequestStatus",
    "SamplingParams",
    # Scheduler
    "Scheduler",
    "SchedulerConfig",
    "SchedulerOutput",
    # Engine
    "EngineCore",
    "AsyncEngineCore",
    "EngineConfig",
    # Model registry
    "get_registry",
    "ModelOwnershipError",
    # Prefix cache (paged SSD-only)
    "BlockAwarePrefixCache",
    # Paged cache (memory efficiency)
    "PagedCacheManager",
    "CacheBlock",
    "BlockTable",
    "PagedCacheStats",
    "CacheStats",  # Backward compatibility alias
    # Version
    "__version__",
    # Metal guard availability (for internal use)
    "_METAL_GUARD_AVAILABLE",
]
