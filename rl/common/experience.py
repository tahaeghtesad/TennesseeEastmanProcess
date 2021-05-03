import numpy as np
from dataclasses import dataclass


@dataclass
class Experience:
    state: np.ndarray
    action: int # na elzaman
    next_state: np.ndarray
    reward: np.float64
    done: bool
