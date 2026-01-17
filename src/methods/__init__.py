"""
CoT Vector methods package.

Available methods:
- ExtractedCoTVector: Extract vectors from activation differences
- LearnableCoTVector: Learn vectors via teacher-student framework
- SelfEvolvedCoTVector: Learn vectors via RL (GRPO/DAPO)
"""

from .base import BaseCoTVectorMethod
from .extracted import ExtractedCoTVector
from .learnable import LearnableCoTVector
from .self_evolved import SelfEvolvedCoTVector

__all__ = [
    "BaseCoTVectorMethod",
    "ExtractedCoTVector",
    "LearnableCoTVector",
    "SelfEvolvedCoTVector",
]