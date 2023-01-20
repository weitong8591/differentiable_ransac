import torch
from utils import *


class EssentialMatrixEstimator(object):
    """
        implementation of Stewenius 5PC algorithm.
    """
    def __init__(self, device='cuda'):
        self.sample_size = 5

    def estimate_model(self, matches, weights=None):
        # minimal solver
        if matches.shape[1] == self.sample_size:
            return self.estimate_minimal_model(matches, weights)

        # non-minial solver
        elif matches.shape[1] > self.sample_size:
            return self.estimate_minimal_model(matches, weights)
        return None

    def estimate_minimal_model(self, pts, weights=None):  # x1 y1 x2 y2
        """
            using 5 points to estimate Essential matrix.
        """
        try:
            pts.shape[1] == self.sample_size
        except Exception as e:
            print(e, "This is not a minimal sample.")

        batch_size, num, _ = pts.shape
        pts1 = pts[:, :, 0:2]
        pts2 = pts[:, :, 2:4]

        # get the points
        x1, y1 = pts1[:, :, 0], pts1[:, :, 1]
        x2, y2 = pts2[:, :, 0], pts2[:, :, 1]

        # Step1: construct the A matrix, A F = 0.
        # 5 equations for 9 variables, A is 5x9 matrix containing epipolar constraints
        # Essential matrix is a linear combination of the 4 vectors spanning the null space of A
        a_59 = torch.ones(x1.shape, device=self.device)
        A_s = torch.stack(
            (torch.mul(x1, x2), torch.mul(x1, y2), x1,
             torch.mul(y1, x2), torch.mul(y1, y2), y1,
             x2, y2, a_59), dim=-1)

        _, _, v = torch.linalg.svd(A_s)#, full_matrices=False) #A_s # eigenvalues in increasing order

        null_space = v[:, -4:, :].transpose(-1, -2)#.float()  # the last four rows

        # use the 4 eigenvectors according to the 4 smallest singular values,
        # E is calculated from 4 basis, E = cx*X + cy*Y + cz*Z + cw*W, up yo common scale = 1
        # X, Y, Z, W = v[:, -1].reshape(3, 3), v[:, -2].reshape(3, 3),
        # v[:, -3].reshape(3, 3), v[:, -4].reshape(3, 3)  # null space
        # null_space_mat= null_space.reshape(3, 3, 4) #X, Y, Z, W
        null_space_mat = null_space.reshape(null_space.shape[0], 3, 3, null_space.shape[-1]).transpose(1, 2)  # X, Y, Z, W
        # Step2: expansion of the constraints:
        # determinant constraint det(E) = 0,
        # trace constraint 2EE^TE - trace(EE^T)E = 0
        constraint_mat = self.get_constraint_mat(null_space_mat, batch_size)

        # Step 3: Eliminate part of the matrix to isolate polynomials in z.
        # solve AX=b
        b = constraint_mat[:, :, 10:]
        eliminated_mat = torch.linalg.solve(constraint_mat[:, :, :10], b)

        action_mat = torch.zeros((batch_size, 10, 10), device=self.device)
        action_mat[:, 0:3] = eliminated_mat[:, 0:3]
        action_mat[:, 3] = eliminated_mat[:, 4]
        action_mat[:, 4] = eliminated_mat[:, 5]
        action_mat[:, 5] = eliminated_mat[:, 7]
        action_mat[:, 6, 0] = -torch.ones_like(action_mat[:, 6, 0])
        action_mat[:, 7, 1] = -torch.ones_like(action_mat[:, 7, 1])
        action_mat[:, 8, 3] = -torch.ones_like(action_mat[:, 8, 3])
        action_mat[:, 9, 6] = -torch.ones_like(action_mat[:, 9, 6])

        ee, vv = torch.linalg.eig(action_mat)

        # put the cx, cy, cz back to get a valid essential matrix
        E_models = null_space.matmul(vv.real[:, -4:])#torch.stack(#.real
        E_models = E_models.transpose(-1, -2).reshape(-1, 3, 3).transpose(-1, -2)

        return E_models

    def get_constraint_mat(self, null_space, B):
        """expansion of the constraints. 10*20"""

        constraint_mat = torch.zeros((B, 10, 20), device=self.device)

        # 1st: trace constraint 2EE^TE - trace(EE^T)E = 0
        # compute the EE^T
        EE_t = torch.zeros((B, 3, 3, 10), device=self.device)

        self.multiply_deg_one_poly(null_space[:, 0, 0], null_space[:, 0, 0])

        for i in range(3):
            for j in range(3):
                EE_t[:, i, j] = 2 * (
                        self.multiply_deg_one_poly(null_space[:, i, 0], null_space[:, j, 0]) + \
                       self.multiply_deg_one_poly(null_space[:, i, 1], null_space[:, j, 1]) + \
                        self.multiply_deg_one_poly(null_space[:, i, 2], null_space[:, j, 2])
                )
        # trace
        trace = EE_t[:, 0, 0] + EE_t[:, 1, 1] + EE_t[:, 2, 2]
        trace_constraint = constraint_mat[:, :9, :]

        # calculate EE^T with E
        for i in range(3):
            for j in range(3):
                trace_constraint[:, 3 * i + j] = self.multiply_two_deg_one_poly(EE_t[:, i, 0], null_space[:, 0, j]) + \
                                                 self.multiply_two_deg_one_poly(EE_t[:, i, 1], null_space[:, 1, j]) + \
                                                 self.multiply_two_deg_one_poly(EE_t[:, i, 2], null_space[:, 2, j]) - \
                                                 0.5 * self.multiply_two_deg_one_poly(trace, null_space[:, i, j])

        # 2nd: singularity constraint det(E) = 0
        det_constraint = self.multiply_two_deg_one_poly(
            self.multiply_deg_one_poly(null_space[:, 0, 1], null_space[:, 1, 2]) -
            self.multiply_deg_one_poly(null_space[:, 0, 2], null_space[:, 1, 1]), null_space[:, 2, 0]) + \
                         self.multiply_two_deg_one_poly(
                             self.multiply_deg_one_poly(null_space[:, 0, 2], null_space[:, 1, 0]) -
                             self.multiply_deg_one_poly(null_space[:, 0, 0], null_space[:, 1, 2]),
                             null_space[:, 2, 1]) + \
                         self.multiply_two_deg_one_poly(
                             self.multiply_deg_one_poly(null_space[:, 0, 0], null_space[:, 1, 1]) -
                             self.multiply_deg_one_poly(null_space[:, 0, 1], null_space[:, 1, 0]),
                             null_space[:, 2, 2])

        # construct the overall constraint 10*20
        constraint_mat[:, :9, :] = trace_constraint
        constraint_mat[:, 9, :] = det_constraint

        return constraint_mat

    def multiply_deg_one_poly(self, a, b):
        """
            from Graph-cut Ransac
            Multiply two degree one polynomials of variables x, y, z.
            E.g. p1 = a[0]x + a[1]y + a[2]z + a[3]
            Output order: x^2 xy y^2 xz yz z^2 x y z 1 ('GrevLex', Graded reverse lexicographic order)
            1*10
            """

        return torch.stack([a[:, 0] * b[:, 0], a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0],
                            a[:, 1] * b[:, 1], a[:, 0] * b[:, 2] + a[:, 2] * b[:, 0],
                            a[:, 1] * b[:, 2] + a[:, 2] * b[:, 1], a[:, 2] * b[:, 2],
                            a[:, 0] * b[:, 3] + a[:, 3] * b[:, 0], a[:, 1] * b[:, 3] + a[:, 3] * b[:, 1],
                            a[:, 2] * b[:, 3] + a[:, 3] * b[:, 2], a[:, 3] * b[:, 3]], dim=-1)


    def multiply_two_deg_one_poly(self, a, b):
        """
        from Graph-cut Ransac
        Multiply two degree one polynomials of variables x, y, z.
        E.g. p1 = a[0]x + a[1]y + a[2]z + a[3]
        Output order: x^2 xy y^2 xz yz z^2 x y z 1 (GrevLex)
        1*20
        """

        return torch.stack([

            a[:, 0] * b[:, 0], a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0], a[:, 1] * b[:, 1] + a[:, 2] * b[:, 0],
            a[:, 2] * b[:, 1], a[:, 0] * b[:, 2] + a[:, 3] * b[:, 0],
            a[:, 1] * b[:, 2] + a[:, 3] * b[:, 1] + a[:, 4] * b[:, 0], a[:, 2] * b[:, 2] + a[:, 4] * b[:, 1],
            a[:, 3] * b[:, 2] + a[:, 5] * b[:, 0],
            a[:, 4] * b[:, 2] + a[:, 5] * b[:, 1], a[:, 5] * b[:, 2],
            a[:, 0] * b[:, 3] + a[:, 6] * b[:, 0], a[:, 1] * b[:, 3] + a[:, 6] * b[:, 1] + a[:, 7] * b[:, 0],
            a[:, 2] * b[:, 3] + a[:, 7] * b[:, 1],
            a[:, 3] * b[:, 3] + a[:, 6] * b[:, 2] + a[:, 8] * b[:, 0],
            a[:, 4] * b[:, 3] + a[:, 7] * b[:, 2] + a[:, 8] * b[:, 1],
            a[:, 5] * b[:, 3] + a[:, 8] * b[:, 2], a[:, 6] * b[:, 3] + a[:, 9] * b[:, 0],
            a[:, 7] * b[:, 3] + a[:, 9] * b[:, 1], a[:, 8] * b[:, 3] + a[:, 9] * b[:, 2],
            a[:, 9] * b[:, 3]

        ], dim=-1)


""" to check if 5PC Stewenius is differentiable. (not)"""
"""
pts = torch.tensor([[[ 5.3496e-03, -5.0421e-02, -3.0159e-02,  1.8618e-02],
         [ 1.5412e-01,  1.6157e-02,  5.9566e-02,  5.6140e-02],
         [ 3.6183e-02,  5.0953e-03, -1.2028e-02,  5.1611e-02],
         [ 3.0706e-02, -1.4839e-01, -1.5853e-02, -5.0327e-02],
         [ 5.1635e-02, -4.2628e-02, -6.9411e-02,  2.7233e-02]]], device='cpu', requires_grad=True)
# pts = torch.tensor([[[0.0859, 0.0412, 0.3694, 0.0248], [0.2730, 0.4039, 0.3167, -0.1500],
#                             [0.1971, 0.4255, 0.2256, 0.1628],
#
#                             [0.1486, -0.2310, 0.1055, 0.2293],
#
#                             [0.0987, -0.4417, -0.4664, 0.2124]]], device='cuda', requires_grad=True)
parser = create_parser(
        description="test 5PC stewenius.")
opt = parser.parse_args()
opt.device='cpu'
e = EssentialMatrixEstimator(opt)
models = e.estimate_model(pts)
models.retain_grad()
target = torch.rand(models.shape, device=models.device)
loss = torch.norm(models-target)#min_matches
try:
    loss.backward()
except Exception as e:
    print(e)
models.grad
"""