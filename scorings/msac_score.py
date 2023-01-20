import torch


class MSACScore(object):

    def __init__(self, threshold, device="cuda"):
        self.threshold = (3 / 2 * threshold)**2
        self.th = (3 / 2) * threshold
        self.device = device
        self.provides_inliers = True

    def score(self, matches, models):
        """
            rewrite from Graph-cut Ransac
            github.com/danini/graph-cut-ransac
            calculate the Sampson distance between a point correspondence and essential/ fundamental matrix.
            Sampson distance is the first order approximation of geometric distance, calculated from the closest correspondence
            who satisfy the F matrix.
            :param: x1: x, y, 1; x2: x', y', 1;
            M: F/E matrix
        """
        pts1 = matches[:, 0:2]
        pts2 = matches[:, 2:4]

        num_pts = pts1.shape[0]
        # truncated_threshold = 3 / 2 * threshold  # wider threshold

        # get homogenous coordinates
        hom_pts1 = torch.cat((pts1, torch.ones((num_pts, 1), device=pts1.device)), dim=-1)
        hom_pts2 = torch.cat((pts2, torch.ones((num_pts, 1), device=pts2.device)), dim=-1)

        # calculate the sampson distance and msac scores
        try:
            M_x1_ = models.matmul(hom_pts1.transpose(-1, -2))
        except:
            print()
        M_x2_ = models.transpose(-1, -2).matmul(hom_pts2.transpose(-1, -2))
        JJ_T_ = M_x1_[:, 0] ** 2 + M_x1_[:, 1] ** 2 + M_x2_[:, 0] ** 2 + M_x2_[:, 1] ** 2

        #x1_M_x2_ = hom_pts1.matmul(M_x2_)
        x1_M_x2_ = hom_pts1.T.unsqueeze(0).mul(M_x2_).sum(-2)

        try:
            # squared_distances = (torch.diagonal(x1_M_x2_, dim1=1, dim2=2)) ** 2 / JJ_T_
            squared_distances = x1_M_x2_.square().div(JJ_T_)
        except Exception as e:
            print("wrong", e)

        masks = squared_distances < self.threshold
        # soft inliers, sum of the squared distance, while transforming the negative ones to zero by torch.clamp()
        msac_scores = torch.sum(torch.clamp(1 - squared_distances / self.threshold, min=0.0), dim=-1)

        # following c++
        #squared_residuals = torch.sum(torch.where(squared_distances>=self.threshold, torch.zeros_like(squared_distances), squared_distances), dim=-1)
        #inlier_number = torch.sum(squared_distances.squeeze(0) < self.threshold, dim=-1)
        # score = (-squared_residuals + inlier_number * self.threshold)/self.threshold

        return msac_scores, masks#, squared_residuals


