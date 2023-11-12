from enum import Enum

class PixelSamplingMethod(Enum):
    EXACT = "EXACT"
    RANDOM = "RANDOM"
    GAUSSIAN = "GAUSSIAN"

class CountSamplingMethod(Enum):
    EXACT = "EXACT"
    VARIABLE = "VARIABLE"
    EXPONENTIAL = "EXPONENTIAL"

