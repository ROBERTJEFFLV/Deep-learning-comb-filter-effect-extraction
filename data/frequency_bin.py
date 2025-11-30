# src/data/frequency_bin.py
class FrequencyBin:
    def __init__(self, freq):
        self.freq = freq
        self.current_amp = 0.0
        self.diff_amp    = 0.0
        self.prev_diff   = 0.0

    def update(self, new_amp, old_amp=None):
        self.current_amp = new_amp
        self.diff_amp    = new_amp - old_amp if old_amp is not None else 0.0

    def second_diff(self):
        d2 = self.diff_amp - self.prev_diff
        self.prev_diff = self.diff_amp
        return d2
