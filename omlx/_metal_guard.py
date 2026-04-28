"""Metal-guard availability flag for internal use across omlx modules."""

# This variable is set by __init__.py during package initialization.
# Other modules import from here rather than from __init__ to avoid
# circular import issues.
_METAL_GUARD_AVAILABLE: bool = False
