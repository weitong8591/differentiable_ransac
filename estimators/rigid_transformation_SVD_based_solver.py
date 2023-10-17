import torch
import numpy as np

class RigidTransformationSVDBasedSolver:
    def __init__(self, data_type=torch.float32, device = 'cuda'):
        self.data_type = data_type
        self.device = device
        self.sample_size = 3
        self.sqrt_3 = torch.sqrt(torch.tensor(3.))

    def estimate_model(self, data, weights=None, sample_indices=None, flag=True):
        """
            https://github.com/danini/graph-cut-ransac/blob/7d4af4d4b3d5e88964631073cfb472921eb118ae/src/pygcransac/include/estimators/solver_rigid_transformation_svd.h#L92
            Now it works for a batch of data, data in a shape of [batch_size, n, 6]
            output: pose in [bs, 4, 3], R, t, scale in batches

        """
        assert data.shape[-1] == 6
        # at least 3 pairs
        assert data.shape[-2] >= 3
        # if the selected indices are given
        if sample_indices is not None:
            points = torch.index_select(data, 0, sample_indices)
        else:
            points = data

        # Calculate the center of gravity for both point clouds
        centroid = torch.mean(points, dim=1)
        coefficient = points - centroid[:, None, :]

        avg_distance0 = torch.sum(torch.sqrt(torch.sum(coefficient[:, :, 0:3] ** 2, dim=-1)), dim=-1) / points.shape[1]
        avg_distance1 = torch.sum(torch.sqrt(torch.sum(coefficient[:, :, 3:6] ** 2, dim=-1)), dim=-1) / points.shape[1]

        coefficients0 = (coefficient.transpose(-1, -2) * weights)[:, 0:3, :] if weights is not None else coefficient.transpose(-1, -2)[:, 0:3, :]
        coefficients1 = (coefficient.transpose(-1, -2) * weights)[:, 3:6, :] if weights is not None else coefficient.transpose(-1, -2)[:, 3:6, :]

        ratio0 = self.sqrt_3 / avg_distance0
        ratio1 = self.sqrt_3 / avg_distance1

        coefficients0 = coefficients0 * ratio0[:, None, None]
        coefficients1 = coefficients1 * ratio1[:, None, None]

        covariance = coefficients0 @ coefficients1.transpose(-1, -2)

        nan_filter = [not torch.isnan(i).any() for i in covariance]
            # covariance = covariance[nan_filter]
        # // A*(f11 f12 ... f33)' = 0 is singular (7 equations for 9 variables), so
        # 				// the solution is linear subspace of dimensionality 2.
        # 				// => use the last two singular std::vectors as a basis of the space
        # 				// (according to SVD properties)
        if flag:
            u, s, v = torch.linalg.svd(covariance.transpose(-1, -2) @ covariance)
        else:
            u, s, v = torch.linalg.svd(covariance.transpose(-1, -2))
        vt = v.clone().transpose(-1, -2)
        R = vt @ u.transpose(-1, -2)

        # singularity
        mask = torch.linalg.det(R) < 0
        if mask.sum() != 0:
            vt[mask, :, 2] = -vt[mask, :, 2]
            R = vt @ u.transpose(-1, -2)

        scale = avg_distance1 / avg_distance0  # no use

        t = torch.sum(R * (-centroid[:, None, 0:3]), dim=1) + centroid[:, 3:6]

        model = torch.cat((
            torch.cat((R, t.unsqueeze(-1)), dim=-1),
            torch.tensor([[0, 0, 0, 1]], device=R.device).repeat(R.shape[0], 1, 1)
            ), dim=1
        )

        return model[nan_filter], R[nan_filter], t[nan_filter], scale[nan_filter]

    def squared_residual(self, pts1, pts2, descriptor, threshold=0.03):
        """
            rewrite from GC-RANSAC,
            https://github.com/danini/graph-cut-ransac/blob/7d4af4d4b3d5e88964631073cfb472921eb118ae/src/pygcransac/include/estimators/rigid_transformation_estimator.h#L162

        """
        assert pts1.shape[1] == 3 # 3D points
        # homogeneous
        pts_t = torch.cat((pts1, torch.ones((pts1.shape[0], 1), dtype=self.data_type, device= pts1.device)), dim=1)

        t = pts_t @ descriptor
        squared_distance = torch.sum((pts2[None, :, :]- t) ** 2, dim=-1)
        inlier_mask = squared_distance < threshold
        return squared_distance.sum(-1), squared_distance.mean(), inlier_mask
