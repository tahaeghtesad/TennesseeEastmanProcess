from typing import List

import numpy as np

from rl.common.experience import Experience
from rl.exceptions.common import EmptyReplayBufferException


class ExperienceReplayBuffer:
    def __init__(self, buffer_size) -> None:
        super().__init__()

        self.buffer_size = buffer_size
        self.buffer: List[Experience] = []

    def sample(self, sample_size):
        if sample_size > len(self.buffer):
            raise EmptyReplayBufferException()
        samples = np.random.choice(self.buffer, sample_size)
        return samples

    def add_experience(self, experience: Experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.buffer_size:
            del self.buffer[0]