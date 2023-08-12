import time
from ransac import RANSAC, RANSAC3D
from estimators.essential_matrix_estimator_nister import *
from estimators.rigid_transformation_SVD_based_solver import  *              
from samplers.uniform_sampler import *
from samplers.gumbel_sampler import *
from scorings.msac_score import *
import torch.nn as nn
import torch.nn.functional as F
from cv_utils import *


def batch_episym(x1, x2, F):
    batch_size, num_pts = x1.shape[0], x1.shape[1]
    x1 = torch.cat([x1, x1.new_ones(batch_size, num_pts, 1)], dim=-1).reshape(batch_size, num_pts, 3, 1)
    x2 = torch.cat([x2, x2.new_ones(batch_size, num_pts, 1)], dim=-1).reshape(batch_size, num_pts, 3, 1)
    F = F.reshape(-1, 1, 3, 3).repeat(1, num_pts, 1, 1)
    x2Fx1 = torch.matmul(x2.transpose(2, 3), torch.matmul(F, x1)).reshape(batch_size, num_pts)
    Fx1 = torch.matmul(F, x1).reshape(batch_size, num_pts, 3)
    Ftx2 = torch.matmul(F.transpose(2, 3), x2).reshape(batch_size, num_pts, 3)

    ys = x2Fx1 ** 2 * (
            1.0 / (Fx1[:, :, 0] ** 2 + Fx1[:, :, 1] ** 2 + 1e-15) +
            1.0 / (Ftx2[:, :, 0] ** 2 + Ftx2[:, :, 1] ** 2 + 1e-15))

    return ys


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)

    return idx[:, :, :]


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx_out = knn(x, k=k)
    else:
        idx_out = idx
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx_out + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class ResNet_Block(nn.Module):
    def __init__(self, inchannel, outchannel, pre=False):
        super(ResNet_Block, self).__init__()
        self.pre = pre
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
        )
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )

    def forward(self, x):
        x1 = self.right(x) if self.pre is True else x
        out = self.left(x)
        out = out + x1
        return torch.relu(out)


class DGCNN_Block(nn.Module):
    def __init__(self, knn_num=9, in_channel=128):
        super(DGCNN_Block, self).__init__()
        self.knn_num = knn_num
        self.in_channel = in_channel

        assert self.knn_num == 9 or self.knn_num == 6
        if self.knn_num == 9:
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channel * 2, self.in_channel, (1, 3), stride=(1, 3)),
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel, self.in_channel, (1, 3)),
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
            )
        if self.knn_num == 6:
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channel * 2, self.in_channel, (1, 3), stride=(1, 3)),
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel, self.in_channel, (1, 2)),
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
            )

    def forward(self, features):
        B, _, N, _ = features.shape
        out = get_graph_feature(features, k=self.knn_num)
        out = self.conv(out)
        return out


class GCN_Block(nn.Module):
    def __init__(self, in_channel):
        super(GCN_Block, self).__init__()
        self.in_channel = in_channel
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.in_channel, (1, 1)),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )

    def attention(self, w):
        w = torch.relu(torch.tanh(w)).unsqueeze(-1)
        A = torch.bmm(w.transpose(1, 2), w)
        return A

    def graph_aggregation(self, x, w):
        B, _, N, _ = x.size()
        with torch.no_grad():
            A = self.attention(w)
            I = torch.eye(N).unsqueeze(0).to(x.device, x.dtype).detach()
            A = A + I
            D_out = torch.sum(A, dim=-1)
            D = (1 / D_out) ** 0.5
            D = torch.diag_embed(D)
            L = torch.bmm(D, A)
            L = torch.bmm(L, D)
        out = x.squeeze(-1).transpose(1, 2).contiguous()
        out = torch.bmm(L, out).unsqueeze(-1)
        out = out.transpose(1, 2).contiguous()

        return out

    def forward(self, x, w):
        out = self.graph_aggregation(x, w)
        out = self.conv(out)
        return out


class RANSACLayer(nn.Module):
    def __init__(self, opt, **kwargs):  
        super(RANSACLayer, self).__init__(**kwargs)
        self.opt = opt
        if opt.precision == 2:
            data_type = torch.float64
        elif opt.precision == 0:
            data_type = torch.float16
        else:
            data_type = torch.float32
        import pdb; pdb.set_trace()
        if opt.fmat:
            # Initialize the fundamental matrix estimator
            solver = FundamentalMatrixEstimatorNew(
                opt.device,
                opt.weighted
                )
        else:
            # Initialize the essential matrix estimator
            solver = EssentialMatrixEstimatorNister(opt.device)

        if self.opt.sampler == 0:
            sampler = UniformSampler(
                opt.ransac_batch_size,
                solver.sample_size,
            )
        elif self.opt.sampler == 1:
            sampler = GumbelSoftmaxSampler(
                opt.ransac_batch_size,
                solver.sample_size,
                device=opt.device,
                data_type=data_type
            )
        elif self.opt.sampler == 2:
            sampler = GumbelSoftmaxSampler(
                opt.ransac_batch_size,
                solver.sample_size,
                device=opt.device,
                data_type=data_type
                )

        else:
            # if self.opt.sampler == 3:
            # 8PC
            sampler = GumbelSoftmaxSampler(
                opt.ransac_batch_size,
                8,
                device=opt.device,
                data_type=data_type
            )

        scoring = MSACScore(self.opt.device)

        # maximal iteration number, fixed when training, adaptive updating while testing
        if opt.fmat:
            # 7PC/8PC
            # and self.opt.sampler == 3:
            max_iters = 1000 if opt.tr else 5000
        else:
            # 5PC
            max_iters = 1000 if opt.tr else 5000

        self.estimator = RANSAC(
            solver,
            sampler,
            scoring,
            max_iterations=max_iters,
            fmat=opt.fmat,
            train=opt.tr,
            ransac_batch_size=opt.ransac_batch_size,
            sampler_id=opt.sampler,
            weighted=opt.weighted,
            threshold=opt.threshold
        )



    def forward(self, points, weights, K1, K2, im_size1, im_size2, ground_truth=None):

        #estimator = self.initialize_ransac(points.shape[0], K1, K2)
        points_ = points.clone()
        if self.opt.fmat:
              points_[:, 0:2] = denormalize_pts(points[:, 0:2].clone(), im_size1)
              points_[:, 2:4] = denormalize_pts(points[:, 2:4].clone(), im_size2)

        start_time = time.time()
        models, _, model_score, iterations = self.estimator(points_, weights, K1, K2, ground_truth)
        ransac_time = time.time() - start_time

        # collect all the models from different iterations
        if self.opt.tr:
            Es = torch.cat(list(models.values()))  # .cpu().detach().numpy() # no gradient again
        else:
            Es = models
        # masks for removing models containing nan values
        nan_filter = [not (torch.isnan(E).any()) for E in Es]

        return Es[nan_filter], ransac_time

# class RANSACLayer(nn.Module):
#     def __init__(self, opt, **kwargs):  # weights,
#         super(RANSACLayer, self).__init__(**kwargs)
#         self.opt = opt

#     def forward(self, points, weights, K1, K2, im_size1, im_size2, ground_truth=None):

#         estimator = self.initialize_ransac(points.shape[0], K1, K2)
#         points_ = points.clone()
#         if self.opt.fmat:
#               points_[:, 0:2] = denormalize_pts(points[:, 0:2].clone(), im_size1)
#               points_[:, 2:4] = denormalize_pts(points[:, 2:4].clone(), im_size2)

#         start_time = time.time()
#         models, _, model_score, iterations = estimator(points_, weights, K1, K2, ground_truth)
#         ransac_time = time.time() - start_time

#         # collect all the models from different iterations
#         if self.opt.tr:
#             Es = torch.cat(list(models.values()))  # .cpu().detach().numpy() # no gradient again
#         else:
#             Es = models
#         # masks for removing models containing nan values
#         nan_filter = [not (torch.isnan(E).any()) for E in Es]

#         return Es[nan_filter], ransac_time

#     def initialize_ransac(self, num_points, K1, K2):
#         if self.opt.precision == 2:
#             data_type = torch.float64
#         elif self.opt.precision == 0:
#             data_type = torch.float16
#         else:
#             data_type = torch.float32

#         if self.opt.fmat:
#             # Initialize the fundamental matrix estimator
#             normalizing_multiplier = 1
#             estimator = FundamentalMatrixEstimatorNew(
#                 self.opt.device,
#                 self.opt.weighted
#                 )
#         else:
#             # Initialize the essential matrix estimator
#             estimator = EssentialMatrixEstimatorNister(self.opt.device)
#             # Normalize the threshold
#             normalizing_multiplier = (K1[0, 0] + K1[1, 1] + K2[0, 0] + K2[1, 1]) / 4

#         if self.opt.sampler == 0:
#             sampler = UniformSampler(
#                 self.opt.ransac_batch_size,
#                 estimator.sample_size,
#                 num_points
#                 )
#         elif self.opt.sampler == 1:
#             sampler = GumbelSoftmaxSampler(
#                 self.opt.ransac_batch_size,
#                 estimator.sample_size,
#                 num_points,
#                 device=self.opt.device,
#                 data_type=data_type
#             )
#         elif self.opt.sampler == 2:
#             # 7PC/ 5PC
#             sampler = GumbelSoftmaxSampler(
#                 self.opt.ransac_batch_size,
#                 estimator.sample_size,
#                 num_points,
#                 device=self.opt.device,
#                 data_type=data_type
#                 )
#         else:
#             # if self.opt.sampler == 3:
#             # 8PC
#             sampler = GumbelSoftmaxSampler(
#                 self.opt.ransac_batch_size,
#                 8,
#                 num_points,
#                 device=self.opt.device,
#                 data_type=data_type
#             )

#         scoring = MSACScore(self.opt.threshold / normalizing_multiplier, self.opt.device)

#         # maximal iteration number, fixed when training, adaptive updating while testing
#         if self.opt.fmat:
#             # 7PC/8PC
#             # and self.opt.sampler == 3:
#             max_iters = 1000 if self.opt.tr else 5000
#         else:
#             # 5PC
#             max_iters = 100 if self.opt.tr else 5000

#         estimator = RANSAC(
#             estimator,
#             sampler,
#             scoring,
#             max_iterations=max_iters,
#             fmat=self.opt.fmat,
#             train=self.opt.tr,
#             ransac_batch_size=self.opt.ransac_batch_size,
#             sampler_id=self.opt.sampler,
#             weighted=self.opt.weighted,
#             threshold=self.opt.threshold / normalizing_multiplier
#         )

#         return estimator


class DS_Block(nn.Module):
    def __init__(self, initial=False, predict=False, out_channel=128, k_num=8, sampling_rate=0.5):
        super(DS_Block, self).__init__()
        self.initial = initial
        self.in_channel = 7
        self.out_channel = out_channel
        self.k_num = k_num
        # flag if we predict the parametric models or only weights
        self.predict = predict
        self.sr = sampling_rate

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, (1, 1)),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )

        self.gcn = GCN_Block(self.out_channel)

        self.embed_0 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            DGCNN_Block(self.k_num, self.out_channel),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )
        self.embed_1 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )
        self.linear_0 = nn.Conv2d(self.out_channel, 1, (1, 1))
        self.linear_1 = nn.Conv2d(self.out_channel, 1, (1, 1))

        if self.predict:
            self.embed_2 = ResNet_Block(self.out_channel, self.out_channel, pre=False)
            self.linear_2 = nn.Conv2d(self.out_channel, 2, (1, 1))

    def down_sampling(self, x, y, weights, indices, features=None, predict=False):
        B, _, N, _ = x.size()
        indices = indices[:, :int(N * self.sr)]
        with torch.no_grad():
            print("y", y.shape)
            print("indices", indices.shape)
            y_out = torch.gather(y, dim=-1, index=indices)
            w_out = torch.gather(weights, dim=-1, index=indices)
        indices = indices.view(B, 1, -1, 1)

        if predict:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))
            return x_out, y_out, w_out
        else:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))
            feature_out = torch.gather(features, dim=2, index=indices.repeat(1, 128, 1, 1))
            return x_out, y_out, w_out, feature_out

    def forward(self, x):
        B, _, N, _ = x.size()
        out = self.conv(x)
        out = self.embed_0(out)
        w0 = self.linear_0(out).view(B, -1)
        out_g = self.gcn(out, w0.detach())
        out = out_g + out
        out = self.embed_1(out)
        w1 = self.linear_1(out).view(B, -1)
        return w1


class DeepRansac_CLNet(nn.Module):
    def __init__(self, opt):
        super(DeepRansac_CLNet, self).__init__()
        self.opt = opt

        # consensus learning layer, to learn inlier probabilities
        self.ds_0 = DS_Block(initial=True, predict=False, out_channel=128, k_num=9, sampling_rate=1.0)

        # custom-layer, Generalized Differentiable RANSAC, to estimate model
        self.ransac_layer = RANSACLayer(opt)

    def forward(self, points, K1, K2, im_size1, im_size2, prob_type=0, gt=None, predict=True):

        B, _, N, _ = points.shape
        w1 = self.ds_0(points)

        if torch.isnan(w1.std()):
            print("output is nan here")
        if torch.isnan(w1).any():
            print(w1)
            raise Exception("the predicted weights are nan")

        log_probs = F.logsigmoid(w1).view(B, -1)
        # normalization in log space such that probabilities sum to 1
        if torch.isnan(log_probs).any():
            print("predicted log probs have nan values")
        # normalizer = torch.logsumexp(log_probs, dim=1)
        # normalizer = normalizer.unsqueeze(1).expand(-1, N)
        # logits = log_probs - normalizer
        # weights = torch.exp(logits)
        weights = torch.exp(log_probs).view(log_probs.shape[0], -1)
        normalized_weights = weights / torch.sum(weights, dim=-1).unsqueeze(-1)

        if prob_type == 0:
            # normalized weights
            output_weights = normalized_weights.clone()
        elif prob_type == 1:
            # unnormalized weights
            output_weights = weights.clone()
        else:
            # logits
            output_weights = log_probs.clone()

        if torch.isnan(output_weights).any():
            # This should never happen! Debug here
            print("nan values in weights", weights)
        ret = []
        avg_time = 0
        if predict:
            for b in range(B):
                if gt is not None:
                    Es, ransac_time = self.ransac_layer(
                        points.squeeze(-1)[b, 0:4].T,
                        output_weights[b],
                        K1[b],
                        K2[b],
                        im_size1[b],
                        im_size2[b],
                        gt[b]
                    )
                else:
                    Es, ransac_time = self.ransac_layer(
                        points.squeeze(-1)[b, 0:4].T,
                        output_weights[b],
                        K1[b],
                        K2[b],
                        im_size1[b],
                        im_size2[b]
                    )

                ret.append(Es)
                avg_time += ransac_time
            return ret, output_weights, avg_time/B
        else:
            return output_weights, avg_time/B


class RANSACLayer3D(nn.Module):
    def __init__(self, opt, **kwargs):  # weights,
        super(RANSACLayer3D, self).__init__(**kwargs)
        self.opt = opt
        if opt.precision == 2:
            data_type = torch.float64
        elif opt.precision == 0:
            data_type = torch.float16
        else:
            data_type = torch.float32

        solver = RigidTransformationSVDBasedSolver()

        if self.opt.sampler == 0:
            sampler = UniformSampler(
                opt.ransac_batch_size,
                solver.sample_size,
            )
        elif self.opt.sampler == 1:
            sampler = GumbelSoftmaxSampler(
                opt.ransac_batch_size,
                solver.sample_size,
                device=opt.device,
                data_type=data_type
            )
        elif self.opt.sampler == 2:
            sampler = GumbelSoftmaxSampler(
                opt.ransac_batch_size,
                solver.sample_size,
                device=opt.device,
                data_type=data_type
                )

        else:
            # if self.opt.sampler == 3:
            # 8PC
            sampler = GumbelSoftmaxSampler(
                opt.ransac_batch_size,
                8,
                device=opt.device,
                data_type=data_type
            )

        scoring = MSACScore(self.opt.device)

        # maximal iteration number, fixed when training, adaptive updating while testing
        max_iters = 1000

        self.estimator = RANSAC3D(
            solver,
            sampler,
            scoring,
            max_iterations=max_iters,
            fmat=opt.fmat,
            train=opt.tr,
            ransac_batch_size=opt.ransac_batch_size,
            sampler_id=opt.sampler,
            weighted=opt.weighted,
            threshold=opt.threshold
        )



    def forward(self, points, weights, ground_truth=None):

        start_time = time.time()
        models, residuals, avg_residuals, model_score, iterations = self.estimator(points, weights, ground_truth)
        ransac_time = time.time() - start_time
        # import pdb; pdb.set_trace()
        # collect all the models from different iterations
        if self.opt.tr:
            Es = torch.cat(list(models.values()))  # .cpu().detach().numpy() # no gradient again
            loss = torch.cat(list(residuals.values()))
            avg_loss = sum(list(avg_residuals.values()))/len(list(avg_residuals.values()))
        else:
            Es = models
        # masks for removing models containing nan values
        nan_filter = [not (torch.isnan(E).any()) for E in Es]

        return Es[nan_filter], loss.mean(), avg_loss, ransac_time

                
                
                
class CLNet(nn.Module):
    def __init__(self):
        super(CLNet, self).__init__()

        # consensus learning layer, to learn inlier probabilities
        self.ds_0 = DS_Block(initial=True, predict=False, out_channel=128, k_num=9, sampling_rate=1.0)


    def forward(self, points, prob_type=0):

        B, _, N, _ = points.shape
        #import pdb; pdb.set_trace()

        w1 = self.ds_0(points)

        if torch.isnan(w1.std()):
            print("output is nan here")
        if torch.isnan(w1).any():
            print(w1)
            raise Exception("the predicted weights are nan")

        log_probs = F.logsigmoid(w1).view(B, -1)
        # normalization in log space such that probabilities sum to 1
        if torch.isnan(log_probs).any():
            print("predicted log probs have nan values")

        weights = torch.exp(log_probs).view(log_probs.shape[0], -1)
        normalized_weights = weights / torch.sum(weights, dim=-1).unsqueeze(-1)

        if prob_type == 0:
            # normalized weights
            output_weights = normalized_weights.clone()
        elif prob_type == 1:
            # unnormalized weights
            output_weights = weights.clone()
        else:
            # logits
            output_weights = log_probs.clone()

        if torch.isnan(output_weights).any():
            # This should never happen! Debug here
            print("nan values in weights", weights)
        return output_weights
