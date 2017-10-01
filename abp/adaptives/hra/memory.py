import numpy as np

class Memory(object):
    """Memory for experience replay"""
    def __init__(self, size):
        self.size = size
        self.memory = []

    @property
    def current_size(self):
        return len(self.memory)

    def add(self, item):
        if len(self.memory) > self.size:
            self.memory.pop()

        self.memory.append(item)

    def sample(self, batch_size):
        if batch_size > len(self.memory):
            return np.random.choice(self.memory, len(self.memory))
        
        return np.random.choice(self.memory, batch_size)
