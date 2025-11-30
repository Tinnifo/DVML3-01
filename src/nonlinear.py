"""
NonLinearModel for evaluation scripts.
This module provides a unified interface for loading NonLinearModel from either OUR.py or REVISIT.py.

The evaluation script (evaluation/utils.py) expects to import NonLinearModel from here.
Both OUR.py and REVISIT.py can be used to train models, and this module will use
the appropriate class based on which one is needed.

Note: OUR.py and REVISIT.py have slightly different architectures:
- OUR.py: linear1 -> batch1 -> dropout1 -> linear2 (no activation)
- REVISIT.py: linear1 -> batch1 -> sigmoid -> dropout1 -> linear2

Models trained with one architecture should be loaded with the matching class.
By default, we use REVISIT's NonLinearModel since it has all required methods.
"""
from src.REVISIT import NonLinearModel as REVISIT_NonLinearModel
from src.OUR import NonLinearModel as OUR_NonLinearModel

# Use REVISIT's NonLinearModel as the default since it has all required methods
# (including read2emb) and is compatible with the evaluation scripts
NonLinearModel = REVISIT_NonLinearModel

__all__ = ["NonLinearModel"]

