"""
CoT Vector methods package.

Available methods:
- ExtractedCoTVector: Extract vectors from activation differences
- LearnableCoTVector: Learn vectors via teacher-student framework
- UncertaintyAwareCoTVector: UA-Vector with Bayesian shrinkage gating
"""

from .base import BaseCoTVectorMethod
from .extracted import ExtractedCoTVector
from .ua_vector import UncertaintyAwareCoTVector

# Method registry mapping method names to classes
METHOD_MAP = {
    "extracted": ExtractedCoTVector,
    "ua_vector": UncertaintyAwareCoTVector,
}

__all__ = [
    "BaseCoTVectorMethod",
    "ExtractedCoTVector",
    "UncertaintyAwareCoTVector",
    "METHOD_MAP",
]