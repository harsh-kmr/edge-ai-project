from collections import deque
class RunningStat:
    def __init__(self, window_size):
        self.n = window_size
        self.count = 0
        self.mean = 0.0
        self.var = 0.0   
        self._buffer = deque(maxlen=window_size)

    def update(self, new_value):
        """Push a new value; if buffer is full, pop the oldest."""
        if self.count < self.n:
            old = 0.0
            self.count += 1
        else:
            old = self._buffer[0]  
        if len(self._buffer) == self.n:
            self._buffer.popleft()
        self._buffer.append(new_value)

        self.mean += (new_value - old) / self.n
        self.var  += (new_value*new_value - old*old) / self.n

    @property
    def variance(self):
        """Returns Var = E[x²] – (E[x])²"""
        return max(self.var - self.mean*self.mean, 0.0)

    @property
    def is_full(self):
        return self.count >= self.n
    
    @property
    def sum(self):
        return self.count * self.mean