"""CARVE: Counterfactual Attribution and Re-grounding of Video Evidence.

The method only - Definitions 1-2 (carve/mask.py, carve/delta.py), the
probe replay engine (carve/probe.py), the re-grounding controller
(carve/controller.py, Section 5), and the agent-wrapping interface
(carve/wrapper.py). Comparison methods live in baselines/, not here.

Deliberately does NOT eagerly import every submodule here: carve.delta has
zero external dependencies (it's the score arithmetic only, see its own
tests), while carve.mask needs Pillow/numpy and carve.probe/controller
need a live agent manager. Importing this package should not force every
dependency on a caller who only wants `from carve import delta`.
"""
__all__ = ['mask', 'delta', 'probe', 'controller', 'wrapper']
__version__ = '0.1.0'
