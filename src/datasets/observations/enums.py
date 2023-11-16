from enum import Enum

class PixelSamplingMethod(Enum):
    EXACT = "EXACT"
    UNIFORM = "UNIFORM"
    GAUSSIAN = "GAUSSIAN"

class CountSamplingMethod(Enum):
    EXACT = "EXACT"
    VARIABLE = "VARIABLE"
    EXPONENTIAL = "EXPONENTIAL"

