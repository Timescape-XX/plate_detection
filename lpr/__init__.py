"""Package init for the tiny CCPD end-to-end licence-plate network."""
from .charset import BLANK_INDEX, CHARSET, NUM_CLASSES, ctc_greedy_decode
from .model import PlateNet

__all__ = ["PlateNet", "CHARSET", "NUM_CLASSES", "BLANK_INDEX", "ctc_greedy_decode"]
