import numpy as np
import pickle

class ReplayBuffer(object):
    def __init__(self, max_len=100000, alpha=1, load_from_file=False):
        self.max_len = max_len
        self.alpha = alpha
        if load_from_file:
            self.buffer = pickle.load(open("buffer.p", "rb"))
            self.weight = pickle.load(open("weight.p", "rb"))
        else:
            self.buffer = []
            # weight is not normalized
            self.weight = np.array([])

    def add(self, episode):
        self.buffer.append(episode)
        self.weight = np.append(self.weight, np.exp(self.alpha*episode['rewards'].sum()))
        if len(self.buffer) > self.max_len:
            delete_ind = np.random.randint(len(self.buffer))
            del self.buffer[delete_ind]
            self.weight = np.delete(self.weight, delete_ind)

    def sample(self):
        return np.random.choice(self.buffer, p=self.weight/self.weight.sum())

    @property
    def trainable(self):
        return len(self.buffer) > 32
