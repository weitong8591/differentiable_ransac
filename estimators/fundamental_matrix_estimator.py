import torch
from utils import *
from math_utils import *


class FundamentalMatrixEstimator(object):

    def __init__(self, device='cuda', weighted=0):
        self.sample_size = 7
        self.device = device
        self.weighted = weighted
        self.eps = 1e-8

    def estimate_model(self, matches, weights=None):
        if matches.shape[1] == self.sample_size:
             return self.estimate_minimal_model(matches, weights)
        elif matches.shape[1] > self.sample_size:
            normalized_matches, T1, T2t = self.normalize(matches)
            return self.estimate_non_minimal_model(normalized_matches, T1, T2t, weights)
        return None

    def normalize(self, matches):
        # The number of points in each minimal sample
        num_points = matches.shape[1]
        # Calculate the mass point for each minimal sample
        mass = torch.mean(matches, dim=1)
        # Substract the mass point of each minimal sample from the corresponding points in both images
        matches = matches - torch.unsqueeze(mass, 1).repeat(1, num_points, 1)
        # Calculate the distances from the mass point for each minimal sample in the source image
        distances1 = torch.linalg.norm(matches[:, :, :2], dim=2)
        # Calculate the distances from the mass point for each minimal sample in the destination image
        distances2 = torch.linalg.norm(matches[:, :, 2:], dim=2)
        # Calculate the average distances in the source image
        avg_distance1 = torch.mean(distances1, dim=1)
        # Calculate the average distances in the destination image
        avg_distance2 = torch.mean(distances2, dim=1)
        # Calculate the scaling to make the average distances sqrt(2) in the source image
        ratio1 = math.sqrt(2) / (avg_distance1 + self.eps)
        # Calculate the scaling to make the average distances sqrt(2) in the destination image
        ratio2 = math.sqrt(2) / (avg_distance2 + self.eps)

        # Calculate the normalized matches in the source image
        normalized_matches1 = matches[:, :, :2] * ratio1.view(-1, 1, 1).repeat(1, num_points, 2)
        # Calculate the normalized matches in the destination image
        normalized_matches2 = matches[:, :, 2:] * ratio2.view(-1, 1, 1).repeat(1, num_points, 2)
        
        # Initialize the normalizing transformations for each minimal sample in the source image
        T1 = torch.zeros((matches.shape[0], 3, 3), device=self.device)
        # Initialize the normalizing transformations for each minimal sample in the destination image
        T2 = torch.zeros((matches.shape[0], 3, 3), device=self.device)

        # Calculate the transformation parameters
        T1[:, 0, 0] = T1[:, 1, 1] = ratio1[:]
        T2[:, 0, 0] = T2[:, 1, 1] = ratio2[:]
        T1[:, 2, 2] = T2[:, 2, 2] = 1 
        T1[:, 0, 2] = -ratio1 * mass[:, 0]
        T1[:, 1, 2] = -ratio1 * mass[:, 1]
        T2[:, 2, 0] = -ratio2 * mass[:, 2]
        T2[:, 2, 1] = -ratio2 * mass[:, 3]
        if torch.isnan(normalized_matches1).any() or torch.isnan(normalized_matches2).any():
            print("normalized_matches contains nans")
        return torch.cat((normalized_matches1, normalized_matches2), dim=2), T1, T2

    def coeff(self, f1, f2):
        # The coefficient calculation for the 7PC algorithm
        fun = lambda a: torch.linalg.det(a * f1 + (1 - a) * f2)
        c0 = fun(0)
        c1 = (fun(1) - fun(-1)) / 3 - (fun(2) - fun(-2)) / 12
        c2 = 0.5 * fun(1) + 0.5 * fun(-1) - fun(0)
        c3 = (fun(1) - fun(-1)) / 6 - (fun(2) - fun(-2)) / 12
        c = torch.stack((c0, c1, c2, c3), dim=-1)
        return c
    
    # Eight-point algorithm
    def estimate_non_minimal_model(self, pts, T1, T2t, weights=None):  # x1 y1 x2 y2
        """
        Using 8 points and singularity constraint to estimate Fundamental matrix.
        """
        # get the points
        B, N, _ = pts.shape
        pts1 = pts[:, :, 0:2]
        pts2 = pts[:, :, 2:4]
        x1, y1 = pts1[:, :, 0], pts1[:, :, 1]
        x2, y2 = pts2[:, :, 0], pts2[:, :, 1]

        a_89 = torch.ones_like(x1)#.shape), device=self.device)

        # construct the A matrix, A F = 0. 8 equations for 9 variables,
        # solution is linear subspace o dimensionality of 2.

        if self.weighted:
            A = weights.unsqueeze(-1) * torch.stack((x1 * x2, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, a_89), dim=-1)#weights.unsqueeze(-1) *
        else:
            A = torch.stack((x1 * x2, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, a_89), dim=-1)

        # solve null space of A to get F
        _, _, v = torch.linalg.svd(A.transpose(-1, -2)@A)#, full_matrices=False)  # eigenvalues in increasing order
        null_space = v[:, -1:, :].transpose(-1, -2).float().clone()  # the last four rows

        # with the singularity constraint, use the last two singular vectors as a basis of the space
        F = null_space[:, :, 0].view(-1, 3, 3)

        if T1 is not None: 
            for i in range(F.shape[0]):
                F[i, :, :] = torch.mm(T2t[i, :, :], torch.mm(F[i, :, :].clone(), T1[i, :, :]))
        if torch.isnan(F).any():
            print("F contains nan values")
        return F

    def estimate_minimal_model(self, pts, weights=None):  # x1 y1 x2 y2
        """ using 7 points and singularity constraint to estimate the Fundamental matrix. """

        # get the points
        B, N, _ = pts.shape
        pts1 = pts[:, :, 0:2]
        pts2 = pts[:, :, 2:4]
        x1, y1 = pts1[:, :, 0], pts1[:, :, 1]
        x2, y2 = pts2[:, :, 0], pts2[:, :, 1]

        a_79 = torch.ones_like(x1)#.shape, device=self.device)

        # construct the A matrix, A F = 0. 7 equations for 9 variables,
        # solution is linear subspace o dimensionality of 2.
        A = torch.stack((x1 * x2, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, a_79), dim=-1)

        # solve null space of A to get F
        _, _, v = torch.linalg.svd(A.transpose(-1, -2)@A)  # eigenvalues in increasing order
        null_space = v[:, -2:, :].transpose(-1, -2).float()  # the last two rows

        # with the singularity constraint, use the last two singular vectors as a basis of the space
        F1 = null_space[:, :, 0].view(-1, 3, 3)
        F2 = null_space[:, :, 1].view(-1, 3, 3)

        # use the two bases, we can have an arbitrary F mat
        # lambda, 1-lambda, det(F) = det(lambda*F1, (1-lambda)*F2) = lambda(F1-F2)+F2 to find lambda
        # c-polynomial coefficients. det(F) = c[0]*lambda^3 + c[1]*lambda^2  + c[2]*lambda + c[3]= 0
        c = self.coeff(F1, F2)

        # solve the cubic equation (1-3 roots)
        s = StrumPolynomialSolverBatch(3, c.shape[0])
        _, roots_ = s.bisect_sturm(c, 3)

        roots = torch.stack(roots_).T

        F_models = []

        for i in range(roots.shape[1]):
            for j in range(3):
                r = roots[j, i]
                lambda_ = r.clone()#.real

                s = F1[i, 2, 2] * r.clone()+ F2[i, 2, 2]#.real
                if torch.abs(s) > 0:
                    # normalize each matrix, F[3,3]=1
                    mu = 1.0 / s
                    lambda_ *= mu
                    F_mat = F1[i, :, :] * lambda_ + F2[i, :, :] * mu
                    F_models.append(F_mat)

        return torch.stack(F_models)
        
        
class FundamentalMatrixEstimatorNew(object):

    def __init__(self, device='cuda', weighted=0):
        self.sample_size = 7
        self.device = device
        self.weighted = weighted
        self.eps = 1e-8

    def estimate_model(self, matches, weights=None):
        if matches.shape[1] == self.sample_size:
             return self.estimate_minimal_model(matches, weights)
        elif matches.shape[1] > self.sample_size:
            normalized_matches, T1, T2t = self.normalize(matches)
            return self.estimate_non_minimal_model(normalized_matches, T1, T2t, weights)
        return None

    def normalize(self, matches):
        dev = matches.device
        # The number of points in each minimal sample
        num_points = matches.shape[1]
        # Calculate the mass point for each minimal sample
        mass = torch.mean(matches, dim=1)
        # Substract the mass point of each minimal sample from the corresponding points in both images
        matches = matches - torch.unsqueeze(mass, 1).repeat(1, num_points, 1)
        # Calculate the distances from the mass point for each minimal sample in the source image
        distances1 = torch.linalg.norm(matches[:, :, :2], dim=2)
        # Calculate the distances from the mass point for each minimal sample in the destination image
        distances2 = torch.linalg.norm(matches[:, :, 2:], dim=2)
        # Calculate the average distances in the source image
        avg_distance1 = torch.mean(distances1, dim=1)
        # Calculate the average distances in the destination image
        avg_distance2 = torch.mean(distances2, dim=1)
        # Calculate the scaling to make the average distances sqrt(2) in the source image
        ratio1 = math.sqrt(2) / avg_distance1
        # Calculate the scaling to make the average distances sqrt(2) in the destination image
        ratio2 = math.sqrt(2) / avg_distance2

        # Calculate the normalized matches in the source image
        normalized_matches1 = matches[:, :, :2] * ratio1.view(-1, 1, 1).repeat(1, num_points, 2)
        # Calculate the normalized matches in the destination image
        normalized_matches2 = matches[:, :, 2:] * ratio2.view(-1, 1, 1).repeat(1, num_points, 2)

        # Initialize the normalizing transformations for each minimal sample in the source image
        T1 = torch.zeros((matches.shape[0], 3, 3), device=dev, dtype=matches.dtype)
        # Initialize the normalizing transformations for each minimal sample in the destination image
        T2 = torch.zeros((matches.shape[0], 3, 3), device=dev, dtype=matches.dtype)

        # Calculate the transformation parameters
        T1[:, 0, 0] = T1[:, 1, 1] = ratio1[:]
        T2[:, 0, 0] = T2[:, 1, 1] = ratio2[:]
        T1[:, 2, 2] = T2[:, 2, 2] = 1
        T1[:, 0, 2] = -ratio1 * mass[:, 0]
        T1[:, 1, 2] = -ratio1 * mass[:, 1]
        T2[:, 2, 0] = -ratio2 * mass[:, 2]
        T2[:, 2, 1] = -ratio2 * mass[:, 3]

        return torch.cat((normalized_matches1, normalized_matches2), dim=2), T1, T2

    def coeff(self, f1, f2):
        # The coefficient calculation for the 7PT algorithm
        fun = lambda a: torch.linalg.det(a * f1 + (1 - a) * f2)
        c0 = fun(0)
        c1 = (fun(1) - fun(-1)) / 3 - (fun(2) - fun(-2)) / 12
        c2 = 0.5 * fun(1) + 0.5 * fun(-1) - fun(0)
        c3 = (fun(1) - fun(-1)) / 6 - (fun(2) - fun(-2)) / 12
        c = torch.stack((c0, c1, c2, c3), dim=-1)
        return c

    # Eight-point algorithm
    def estimate_non_minimal_model(self, pts, T1, T2t, weights=None):  # x1 y1 x2 y2
        """ Using 8 points and singularity constraint to estimate Fundamental matrix. """
        # get the points
        B, N, _ = pts.shape
        pts1 = pts[:, :, 0:2]
        pts2 = pts[:, :, 2:4]
        x1, y1 = pts1[:, :, 0], pts1[:, :, 1]
        x2, y2 = pts2[:, :, 0], pts2[:, :, 1]

        a_89 = torch.ones_like(x1)#.shape, device=pts.device)

        # construct the A matrix, A F = 0. 8 equations for 9 variables,
        # solution is the linear subspace o dimensionality of 2.
        if weights is not None:
            A = weights.unsqueeze(-1) * torch.stack((x1 * x2, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, a_89), dim=-1)#weights.unsqueeze(-1) *
        else:
            A = torch.stack((x1 * x2, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, a_89), dim=-1)

        # solve null space of A to get F
        _, _, v = torch.linalg.svd(A.transpose(-1, -2)@A)#, full_matrices=False)  # eigenvalues in increasing order
            
        null_space = v[:, -1:, :].transpose(-1, -2).to(v.dtype).clone()  # the last four rows

        # with the singularity constraint, use the last two singular vectors as a basis of the space
        F = null_space[:, :, 0].view(-1, 3, 3)

        if T1 is not None:
            for i in range(F.shape[0]):
                F[i, :, :] = torch.mm(T2t[i, :, :], torch.mm(F[i, :, :].clone(), T1[i, :, :]))

        return F

    def estimate_minimal_model(self, pts, weights=None):  # x1 y1 x2 y2
        """ using 7 points and singularity constraint to estimate Fundamental matrix. """

        # get the points
        B, N, _ = pts.shape
        pts1 = pts[:, :, 0:2]
        pts2 = pts[:, :, 2:4]
        x1, y1 = pts1[:, :, 0], pts1[:, :, 1]
        x2, y2 = pts2[:, :, 0], pts2[:, :, 1]

        a_79 = torch.ones_like(x1)#.shape, device=pts.device)

        # construct the A matrix, A F = 0. 7 equations for 9 variables,
        # solution is linear subspace o dimensionality of 2.
        A = torch.stack((x1 * x2, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, a_79), dim=-1)

        # solve null space of A to get F
        _, _, v = torch.linalg.svd(A.transpose(-1, -2)@A)  # eigenvalues in increasing order
        null_space = v[:, -2:, :].transpose(-1, -2)  # the last two rows

        # with the singularity constraint, use the last two singular vectors as a basis of the space
        F1 = null_space[:, :, 0].view(-1, 1, 3, 3)
        F2 = null_space[:, :, 1].view(-1, 1, 3, 3)

        # use the two bases, we can have an arbitrary F mat
        # lambda, 1-lambda, det(F) = det(lambda*F1, (1-lambda)*F2) = lambda(F1-F2)+F2 to find lambda
        # c-polynomial coefficients. det(F) = c[0]*lambda^3 + c[1]*lambda^2  + c[2]*lambda + c[3]= 0
        c = self.coeff(F1, F2).squeeze()

        compmat = torch.zeros((c.shape[0], 4, 4), dtype=c.dtype, device=c.device)
        compmat[:, 1, 0] = 1.
        compmat[:, 2, 1] = 1.
        compmat[:, 3, 2] = 1.
        compmat[..., 2] = -c
        vv = torch.linalg.eigvals(compmat)

        roots = vv.real
        s = F1[..., 2, 2] * roots + F2[..., 2, 2]
        valid_mask = (s > 1e-10) & (vv.imag.abs() < 1e-9)
        mu = 1.0 / s
        lambda_ = roots * mu
        Fs = F1 * lambda_.view(-1, 4, 1, 1) + F2 * mu.view(-1, 4, 1, 1)
        # Fs = Fs[valid_mask]#.view(-1, 3, 3)
        # keep the dim for selecting the best of each four solutions during training
        # fill the invalid ones with identity matrices instead of slicing with the valid mask
        Fs[~valid_mask] = torch.eye(3, device=Fs.device, dtype=Fs.dtype).repeat(valid_mask.shape[0] * 4 - torch.sum(valid_mask), 1, 1)
        return Fs.view(-1, 3, 3)



# unit test
#
# batch_size = 4
# num_points = 10
# num_samples = 7
# coordinates = 4
# parser = create_parser(
#         description="test 8PC.")
# pts = torch.tensor([[[ 5.3496e-03, -5.0421e-02, -3.0159e-02,  1.8618e-02],
#         [ 1.5412e-01,  1.6157e-02,  5.9566e-02,  5.6140e-02],
#         [ 3.6183e-02,  5.0953e-03, -1.2028e-02,  5.1611e-02],
#         [ 3.0706e-02, -1.4839e-01, -1.5853e-02, -5.0327e-02],
#         [ 5.1635e-02, -4.2628e-02, -6.9411e-02,  2.7233e-02],
#                     [ 5.1635e-02, -4.2628e-02, -6.9411e-02,  2.7233e-02],
#                     [ 5.1635e-02, -4.2628e-02, -6.9411e-02,  2.7233e-02]]], device='cuda:0', requires_grad=True)
# estimator = FundamentalMatrixEstimatorNew()
# matches = torch.rand([batch_size, num_samples, coordinates], device='cuda:0')
# matches.requires_grad = True
# estimator_old = FundamentalMatrixEstimator()
# m = estimator_old.estimate_model(pts)
# models = estimator.estimate_model(pts)
# models.retain_grad()
# target = torch.rand(models.shape, device=models.device)
# loss = torch.norm(models-target)#min_matches
# try:
#     loss.backward()
#     print("successfully")
# except Exception as e:
#     print(e)


