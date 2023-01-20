import cv2
import torch
import torch.nn as nn
from cv_utils import *
from math_utils import *
import numpy as np
from scorings.msac_score import *
from feature_utils import *


class PoseLoss(nn.modules.Module):
    """average rotation and translation errors returned, w0"""

    def __init__(self, fmat=False):
        self.fmat = fmat

    def forward_average(
            self,
            estimated_models,
            pts1,
            pts2,
            gt_R,
            gt_t,
            K1=None,
            K2=None,
            im_size1=None,
            im_size2=None
    ):

        train_loss = []
        for b, models in enumerate(estimated_models):

            if self.fmat:
                # E = K2^-T @ F @ K1
                Es = K2[b].transpose(-1, -2) @ estimated_models[b] @ K1[b]

                # recover the original pixel size and normalize using K
                pts1_1 = normalize_keypoints_tensor(
                    denormalize_pts(pts1[b].clone(), im_size1[b]),
                    K1[b]
                ).cpu().detach().numpy()

                pts2_2 = normalize_keypoints_tensor(
                    denormalize_pts(
                        pts2[b].clone(), im_size2[b]),
                    K2[b]
                ).cpu().detach().numpy()

            else:
                # E
                pts1_1 = pts1[b].clone().cpu().detach().numpy()
                pts2_2 = pts2[b].clone().cpu().detach().numpy()
                Es = estimated_models[b]

            # recover the motions, calculate the epipolar error
            # Rs, ts = motion_from_essential_choose_solution(models, K1[b].repeat(models.shape[0], 1, 1), K2[b].repeat(models.shape[0], 1, 1), pts1[b].repeat(models.shape[0], 1, 1), pts2[b].repeat(models.shape[0], 1, 1))
            loss = []
            for i in range(Es.shape[0]):
                # pose error calculation
                errR, errT = eval_essential_matrix(pts1_1, pts2_2, models[i], gt_R[b], gt_t[b])
                try:
                    loss.append((errR + errT) / 2) # average
                except:
                    loss.append(torch.tensor(0.0, device=models.device))
            train_loss.append(sum(loss) / models.shape[0])

        return sum(train_loss) / gt_R.shape[0]


class ClassificationLoss(nn.modules.Module):
    """return the binary classification loss, w1"""

    def __init__(self, fmat):
        self.fmat = fmat

    def forward(self, gt_E, pts1, pts2, logits, K1, K2, im_size1, im_size2):
        loss_fun = torch.nn.BCELoss()
        gt_masks = []

        for b, l in enumerate(logits):
            if self.fmat:
                pts1_1 = cv2.undistortPoints(
                    denormalize_pts(pts1[b].clone().unsqueeze(0), im_size1[b]).cpu().detach().numpy(),
                    K1[b],
                    None
                )
                pts2_2 = cv2.undistortPoints(
                    denormalize_pts(pts2[b].clone().unsqueeze(0), im_size2[b]).cpu().detach().numpy(),
                    K2[b],
                    None
                )

            else:
                pts1_1 = pts1[b].cpu().detach().numpy()
                pts2_2 = pts2[b].cpu().detach().numpy()

            _, gt_R_1, gt_t_1, gt_inliers = cv2.recoverPose(gt_E[b].astype(np.float64), pts1_1, pts2_2, np.eye(3))

            gt_mask = np.where(gt_inliers.ravel() > 0, 1.0, 0.0)
            gt_mask = torch.from_numpy(gt_mask).to(l.device, l.dtype)
            gt_masks.append(gt_mask)
            # TODO: calculate gt probabilities \in (0, 1) instead of gt masks \in [0, 1]
        return loss_fun(logits, torch.stack(gt_masks))


class MatchLoss(object):
    """rewrite Match loss from CLNet, symmetric epipolar error, w3"""

    def __init__(self, fmat):
        self.scoring_fun = MSACScore(fmat)
        self.fmat = fmat

    def forward(self, models, gt_E, pts1, pts2, K1, K2, im_size1, im_size2, best_flag=False):
        essential_loss = []
        for b in range(gt_E.shape[0]):
            if self.fmat:
                Es = K2[b].transpose(-1, -2) @ models[b] @ K1[b]
                pts1_1 = normalize_keypoints_tensor(denormalize_pts(pts1[b].clone(), im_size1[b]), K1[b])
                pts2_2 = normalize_keypoints_tensor(denormalize_pts(pts2[b].clone(), im_size2[b]), K2[b])
            else:
                pts1_1 = pts1[b].clone()
                pts2_2 = pts2[b].clone()
                Es = models[b]

            _, gt_R_1, gt_t_1, gt_inliers = cv2.recoverPose(
                gt_E[b].astype(np.float64),
                pts1_1.unsqueeze(1).cpu().detach().numpy(),
                pts2_2.unsqueeze(1).cpu().detach().numpy(),
                np.eye(3, dtype=gt_E.dtype)
            )

            # find the ground truth inliers
            gt_mask = np.where(gt_inliers.ravel() > 0, 1.0, 0.0).astype(np.bool)
            gt_mask = torch.from_numpy(gt_mask).to(pts1_1.device)

            # symmetric epipolar errors based on gt inliers
            geod = batch_episym(
                pts1_1[gt_mask].repeat(Es.shape[0], 1, 1),
                pts2_2[gt_mask].repeat(Es.shape[0], 1, 1),
                Es
            )
            e_l = torch.min(geod, geod.new_ones(geod.shape))
            if torch.isnan(e_l.mean()).any():
                print("nan values in pose loss")# .1*
            if best_flag:
                essential_loss.append(torch.min(e_l))
            else:
                essential_loss.append(e_l.mean())

        # average
        return sum(essential_loss) / gt_E.shape[0]

    # def msac_loss(self, models, pts1, pts2, K1, K2, im_size1, im_size2):
    #     # losses = []
    #     msac_score = []
    #     for b in range(len(models)):
    #         if self.fmat:
    #             pts1_1 = denormalize_pts(pts1[b].clone(), im_size1[b])
    #             pts2_2 = denormalize_pts(pts2[b].clone(), im_size2[b])
    #         scores, _, _ = self.scoring_fun.score(torch.concat((pts1_1, pts2_2), dim=-1), models[b])  # _single
    #
    #         msac_score.append(torch.mean(scores) / 2000)
    #         # losses.append()
    #
    #     return 1.0 - sum(msac_score) / len(models)
