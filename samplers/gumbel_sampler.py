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
    def __init__(self, batch_size, num_samples, num_points, tau=1., device='cuda', data_type='torch.float32'):
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.num_points = num_points
        self.device = device
        self.dtype = data_type
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(
                torch.tensor(0., device=self.device, dtype=self.dtype),
                torch.tensor(1., device=self.device, dtype=self.dtype))
        self.tau = tau

    def sample(self, logits=None, selected=None):

        if logits==None:
            logits = torch.ones([self.batch_size, self.num_points], device=self.device, dtype=self.dtype, requires_grad=True)
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

        return ret, y_soft


# Unit test to see if grads e
"""
batch_size = 4
num_points = 10
num_samples = 5
coordinates = 4

loss_fn = PoseLoss()
estimator = FundamentalMatrixEstimator(device='cpu')
criterion = torch.nn.MSELoss()
target = torch.rand([batch_size, num_samples])
#6
matches = torch.rand([num_points, coordinates])#
matches.requires_grad = True
matches_ = matches.repeat([batch_size, 1, 1])
#indices = torch.range(0, matches.shape[1]-1) # 4*10 index

matches_.retain_grad()

sampler = GumbelSoftmaxSampler(batch_size, num_samples, num_points, 'cpu')
#logits = torch.ones([batch_size, num_points])
#logits.requires_grad = True
samples = sampler.sample()#logits
samples.retain_grad()
min_matches = matches_[samples != 0].view(batch_size, num_samples, -1)
min_matches.retain_grad()#
#target = torch.rand(min_matches.shape)
models = estimator.estimate_model(min_matches)#
models.retain_grad()
target = torch.rand(models.shape)
#loss = loss_fn.forward(models, target, np.random.rand(2000, 2), np.random.rand(2000, 2), target, torch.rand([3,1]))#torch.norm(models-target)

loss = torch.norm(models-target)#min_matches
loss.retain_grad()
loss.backward()
matches.grad

#
# with torch.autograd.set_detect_anomaly(True):
#     min = torch.rand([batch_size, num_samples, 4])#
#     min.requires_grad = True
#     models = estimator.estimate_model(min)#
#     models.retain_grad()
#     target = torch.rand(models.shape)
#     loss = torch.norm(models-target)#min_matches
#     loss.backward()
#     matches.grad
"""