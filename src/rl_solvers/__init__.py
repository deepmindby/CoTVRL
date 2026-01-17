"""
RL Solvers for Self-Evolved Task Vector Training.

This package contains implementations of:
- GRPO (Group Relative Policy Optimization)
- DAPO (Direct Alignment Policy Optimization)
"""

from .grpo import GRPOSolver
from .dapo import DAPOSolver

__all__ = ["GRPOSolver", "DAPOSolver"]