from torch import Tensor
from typing import List, Tuple

VAEModelOutput = Tuple[List[Tensor], List[Tensor | None], Tensor, Tensor, Tensor]