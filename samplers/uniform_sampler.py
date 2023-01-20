import torch
import random

class UniformSampler(object):
    """
        random sampling the points, return the indices for each unique subset, or in batch.
    """
    def __init__(self, batch_size, num_samples, num_points):
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.num_points = num_points

    def unique_generate(self, num_points):
        sample_indices = torch.tensor([random.randint(0, len(num_points) - 1) for _ in range(self.num_samples)])
        return sample_indices

    def batch_generate(self):
        sample_indices = torch.randint(0, 
            self.num_points - 1, 
            (self.batch_size, self.num_samples))
        return sample_indices

    def sample(self):
        sample_indices = self.batch_generate()
        return sample_indices
