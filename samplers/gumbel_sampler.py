import torch
from estimators.fundamental_matrix_estimator import *
from estimators.essential_matrix_estimator_stewenius import *
from loss import *
import numpy as np
import random


class GumbelSoftmaxSampler():
    '''
        Sample based on a Gumbel-Max distribution.
        Use re-param trick for back-prop
    '''
    def __init__(self, batch_size, num_samples, tau=1., device='cuda', data_type='torch.float32'):
        self.batch_size = batch_size
        self.num_samples = num_samples
        # self.num_points = num_points
        self.device = device
        self.dtype = data_type
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(
                torch.tensor(0., device=self.device, dtype=self.dtype),
                torch.tensor(1., device=self.device, dtype=self.dtype))
        self.tau = tau

    def sample(self, logits=None, num_points=2000, selected=None):

        if logits==None:
            logits = torch.ones([self.batch_size, num_points], device=self.device, dtype=self.dtype, requires_grad=True)
        else:
            logits = logits.to(self.dtype).to(self.device).repeat([self.batch_size, 1])

        if selected is None:
            gumbels = self.gumbel_dist.sample(logits.shape)
            gumbels = (logits + gumbels)/self.tau
            y_soft = gumbels.softmax(-1)
            topk = torch.topk(gumbels, self.num_samples, dim=-1)
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(-1, topk.indices, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            pass

        return ret, y_soft#, topk.indices
