# Make the repo root importable so `import dimer_transport` and related worker
# modules resolve under the `pytest` console script (which, unlike
# `python -m pytest`, does not reliably add the cwd to sys.path).
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
