import cv2
import math
import torch
import numpy as np
#import kornia
# from kornia.geometry.epipolar import motion_from_essential
# from kornia.geometry.epipolar.projection import projection_from_KRt, depth_from_point
# from kornia.geometry.epipolar.triangulation import triangulate_points
# from kornia.utils import eye_like, vec_like
#evaluate_R_t_tensor


def normalize_pts(pts, im_size):
    """Normalize image coordinate using the image size.

	Pre-processing of correspondences before passing them to the network to be
	independent of image resolution.
	Re-scales points such that max image dimension goes from -0.5 to 0.5.
	In-place operation.

	Keyword arguments:
	pts -- 3-dim array conainting x and y coordinates in the last dimension, first dimension should have size 1.
	im_size -- image height and width
	"""

    #pts[0, :, 0] -= float(im_size[1]) / 2
    #pts[0, :, 1] -= float(im_size[0]) / 2
    #pts /= float(max(im_size))
    ret = pts.clone() / max(im_size) - torch.stack((im_size[1]/2, im_size[0]/2))
    return ret


def denormalize_pts_inplace(pts, im_size):
    """Undo image coordinate normalization using the image size.
        Keyword arguments:
            pts -- N-dim array conainting x and y coordinates in the first dimension
            im_size -- image height and width
    """
    pts *= max(im_size)
    pts[0] += im_size[1] / 2
    pts[1] += im_size[0] / 2


def denormalize_pts(pts, im_size):
    """Undo image coordinate normalization using the image size.
        Keyword arguments:
            pts -- N-dim array containing x and y coordinates in the first dimension
            im_size -- image height and width
    """
    #ret = pts * max(im_size)
    #ret[:, 0] += im_size[1] / 2
    #ret[:, 1] += im_size[0] / 2
    ret = pts.clone() * max(im_size) + torch.stack((im_size[1]/2, im_size[0]/2))

    return ret


def recoverPose(model, p1, p2, distanceThreshold=50):
    """ recover the relative poses (R, t) from essential matrix, and choose the correct solution from 4"""

    # decompose E matirx to get R1, R2, t, -t
    R1, R2, t = decompose_E(model)

    # four solutions
    P = []
    P.append(torch.eye(3, 4, device=model.device, dtype=model.dtype))
    P.append(torch.cat((R1, t), 1))
    P.append(torch.cat((R2, t), 1))
    P.append(torch.cat((R1, -t), 1))
    P.append(torch.cat((R2, -t), 1))

    # cheirality check
    mask = torch.zeros(4, p1.shape[0])
    for i in range(len(P) - 1):
        mask[i] = cheirality_check(P[0], P[i + 1], p1, p2, distanceThreshold)

    good = torch.sum(mask, dim=1)
    best_index = torch.argmax(good)

    if best_index == 0:
        return R1, t
    elif best_index == 1:
        return R2, t
    elif best_index == 2:
        return R1, -t
    else:
        return R2, -t


def decompose_E(model):
    try:
        u, s, vT = torch.linalg.svd(model)
        #_, vT = torch.linalg.eig(model.transpose(-1, -2) @model)
        #_, u = torch.linalg.eig(model@model.transpose(-1, -2))
    except Exception as e:
        print(e)
        model = torch.eye(3, device=model.device)
        u, s, vT = torch.linalg.svd(model)

    try:
        if torch.sum(torch.isnan(u)) >0:
            print("wrong")
    except Exception as e:
        print(e)

    w = torch.tensor([[0, -1, 0],
                      [1, 0, 0],
                      [0, 0, 1]], dtype=u.dtype, device=u.device)
    z = torch.tensor([[0, -1, 0],
                      [1, 0, 0],
                      [0, 0, 0]], dtype=u.dtype, device=u.device)

    # u_ = u.real.clone() * (-1.0) if torch.det(u.real.clone()) < 0 else u.real.clone()
    #
    # vT_ = vT.real.clone() * (-1.0) if torch.det(vT.real.clone()) < 0 else vT.real.clone()

    u_ = u * (-1.0) if torch.det(u) < 0 else u

    vT_ = vT * (-1.0) if torch.det(vT) < 0 else vT

    R1 = u_ @ w @ vT_

    R2 = u_ @ w.transpose(0, 1) @ vT_

    t = u[:, -1]#real

    return R1, R2, t.unsqueeze(1)


def cheirality_check(P0, P, p1, p2, distanceThreshold):
    #Q = kornia.geometry.epipolar.triangulate_points(P0.repeat(1024, 1, 1), P, p1, p2)
    # make sure the P type, complex tensor with cause error here
    Q = torch.tensor(cv2.triangulatePoints(P0.cpu().detach().numpy(), P.cpu().detach().numpy(), p1.T, p2.T),
                     dtype=P0.dtype, device=P0.device)
    Q_homogeneous = torch.stack([Q[i] / Q[-1] for i in range(Q.shape[0])])
    Q_ = P @ Q_homogeneous
    mask = (Q[2].mul(Q[3]) > 0) & (Q_homogeneous[2] < distanceThreshold) & (Q_[2] > 0) & (Q_[2] < distanceThreshold)

    return mask


def quaternion_from_matrix(matrix, isprecise=False):
    '''Return quaternion from rotation matrix.
    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.
    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> numpy.allclose(quaternion_from_matrix(R, isprecise=False),
    ...                quaternion_from_matrix(R, isprecise=True))
    True
    '''

    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]

        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                      [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                      [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                      [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0

        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0.0:
        np.negative(q, q)

    return q


def quaternion_from_matrix_tensor(matrix, isprecise=False):
    '''Return quaternion from rotation matrix.
    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.
    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> numpy.allclose(quaternion_from_matrix(R, isprecise=False),
    ...                quaternion_from_matrix(R, isprecise=True))
    True
    '''

    #M = torch.tensor(matrix, dtype=torch.float64, device=matrix.device)[:4, :4]
    M = matrix
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]

        # symmetric matrix K
        K = torch.tensor([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                      [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                      [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                      [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]], device=matrix.device)
        K /= 3.0

        # quaternion is an eigenvector of K that corresponds to the largest eigenvalue
        w, V = torch.linalg.eigh(K)
        q = V[[3, 0, 1, 2], torch.argmax(w)]

    if q[0] < 0.0:
        torch.negative(q)

    return q


def evaluate_R_t_tensor(R_gt, t_gt, R, t, q_gt=None):
    t = t.flatten()
    t_gt = t_gt.flatten().to(t.device)

    eps = torch.tensor(1e-8).to(t.device)
    err_q = torch.arccos(torch.max(torch.min((torch.trace(R @ R_gt.transpose(0, 1)) - 1) * 0.5,
                                 torch.tensor(1.0, device=R.device)), torch.tensor(-1.0, device=R.device)))

    t_gt_ = t_gt / (torch.linalg.norm(t_gt) + eps)
    loss_t = torch.max(eps, 1.0 - torch.sum(t * t_gt_)**2)#torch.clamp((), min=eps)
    err_t = torch.arccos(torch.sqrt(1 - loss_t+eps))

    if torch.sum(torch.isnan(err_q)) or torch.sum(torch.isnan(err_t)):
        print("This should never happen! Debug here", R_gt, t_gt, R, t, q_gt)
        import IPython
        IPython.embed()

    return err_q, err_t


def evaluate_R_t_tensor_batch(R_gt, t_gt, R, t, q_gt=None):
    t = t.flatten(-2, -1)
    t_gt = t_gt.flatten().to(t.device)

    eps = torch.tensor(1e-15).to(t.device)
    # rotation error
    err_q = torch.arccos(torch.clamp((torch.diagonal(R @ R_gt.repeat(R.shape[0], 1, 1).transpose(-2, -1), dim1=-2, dim2=-1).sum(-1) - 1) * 0.5, min= -1.0, max=1.0))
    # translation error
    t = t / (torch.norm(t, dim=-1) + eps).unsqueeze(-1)
    t_gt_ = t_gt / (torch.linalg.norm(t_gt) + eps)
    loss_t = torch.clamp((1.0 - torch.sum(t * t_gt_, dim=-1)**2), min=eps)
    err_t = torch.arccos(torch.sqrt(1 - loss_t + 1e-8))

    if torch.sum(torch.isnan(err_q)) or torch.sum(torch.isnan(err_t)):
        # This should never happen! Debug here
        print(R_gt, t_gt, R, t, q_gt)
        import IPython
        IPython.embed()

    return err_q, err_t


def evaluate_R_t(R_gt, t_gt, R, t, q_gt=None):
    t = t.flatten()
    t_gt = t_gt.flatten()

    eps = 1e-15

    if q_gt is None:
        q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt)**2))
    err_q = np.arccos(1 - 2 * loss_q)

    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt)**2))
    err_t = np.arccos(np.sqrt(1 - loss_t))

    if np.sum(np.isnan(err_q)) or np.sum(np.isnan(err_t)):
        print("This should never happen! Debug here", R_gt, t_gt, R, t, q_gt)
        import IPython
        IPython.embed()

    return err_q, err_t


def orientation_error(pts1, pts2, M, ang):
    """orientation error calculation for E or F matrix"""
    # 2D coordinates to 3D homogeneous coordinates

    num_pts = pts1.shape[0]

    # get homogenous coordinates
    hom_pts1 = torch.cat((pts1, torch.ones((num_pts, 1), device=M.device)), dim=-1)
    hom_pts2 = torch.cat((pts2, torch.ones((num_pts, 1), device=M.device)), dim=-1)

    # calculate the ang between n1 and n2
    l1 = M.transpose(-2, -1)@hom_pts2.transpose(-2, -1)[0:2]
    l2 = M@hom_pts1.transpose(-2, -1)[0:2]
    n1 = l1[:, 0:2, :]
    n2 = l2[:, 0:2, :]

    n1_norm = 1 / torch.norm(n1, axis=0)
    n1 = torch.dot(n1, n1_norm)

    n2_norm = 1 / torch.norm(n2, axis=0)
    n2 = torch.dot(n2, n2_norm)

    alpha = torch.arccos(n1.T.dot(n2))

    ori_error = abs(alpha - ang[b])

    return ori_error


def scale_error(pts1, pts2, M, scale_ratio):
    """scale error of the essential/ fundamental matrix"""

    num_pts = pts1.shape[0]

    # get homogeneous coordinates
    hom_pts1 = torch.cat((pts1, torch.ones((num_pts, 1), device=M.device)), dim=-1)
    hom_pts2 = torch.cat((pts2, torch.ones((num_pts, 1), device=M.device)), dim=-1)

    # calculate the angle between n1 and n2
    l1 = (M.transpose(-2, -1) @ (hom_pts2.transpose(-2, -1)))[:, 0:2]
    l2 = (M@(hom_pts1.transpose(-2, -1)))[:, 0:2]

    l1_norm = torch.norm(scale_ratio*l1, dim=(-1, -2))
    l2_norm = torch.norm(l2, dim=(-1, -2))

    return abs(l1_norm - l2_norm)


def eval_essential_matrix_numpy(p1n, p2n, E, dR, dt):
    """recover the rotation and translation matrices through OpneCV and return their errors"""

    if len(p1n) != len(p2n):
        raise RuntimeError('Size mismatch in the keypoint lists')

    if p1n.shape[0] < 5:
        return np.pi, np.pi / 2

    if E is not None:#.size > 0:
        _, R, t, _ = cv2.recoverPose(E.cpu().detach().numpy().astype(np.float64), p1n, p2n)
        #R, t = recoverPose(E, p1n, p2n)
        try:
            err_q, err_t = evaluate_R_t(dR, dt, R, t)
        except:
            err_q = np.pi
            err_t = np.pi / 2

    else:
        err_q = np.pi
        err_t = np.pi / 2

    return err_q / np.pi * 180.0, err_t / np.pi * 180.0


def eval_essential_matrix(p1n, p2n, E, dR, dt):
    """evaluate the essential matrix, decompose E to R and t, return the rotation and translation error."""

    if len(p1n) != len(p2n):
        raise RuntimeError('Size mismatch in the keypoint lists')

    if p1n.shape[0] < 5:
        return np.pi, np.pi / 2

    if E is not None:
        # recover the relative pose from E
        R, t = recoverPose(E, p1n, p2n)
        try:
            err_q, err_t = evaluate_R_t_tensor(dR, dt, R, t)
        except:
            err_q = np.pi
            err_t = np.pi / 2

    else:
        err_q = np.pi
        err_t = np.pi / 2

    return err_q / np.pi * 180.0, err_t / np.pi * 180.0


def AUC(losses, thresholds=[5, 10, 20], binsize=5):
    """
    from NG-RANSAC
    Compute the AUC up to a set of error thresholds.

    Return multiple AUC corresponding to multiple threshold provided.
    Keyword arguments:
    losses -- list of losses which the AUC should be calculated for
    thresholds -- list of threshold values up to which the AUC should be calculated
    binsize -- bin size to be used fo the cumulative histogram when calculating the AUC, the finer the more accurate
    """

    bin_num = int(max(thresholds) / binsize)
    bins = np.arange(bin_num + 1) * binsize

    hist, _ = np.histogram(losses, bins)  # histogram up to the max threshold
    hist = hist.astype(np.float32) / len(losses)  # normalized histogram
    hist = np.cumsum(hist)  # cumulative normalized histogram

    # calculate AUC for each threshold
    return [np.mean(hist[:int(t / binsize)]) for t in thresholds]


def AUC_tensor(losses, thresholds=[5, 10, 20], binsize=5):
    """
        re-implementation in PyTorch from NG-RANSAC
        Compute the AUC up to a set of error thresholds.

        Return multiple AUC corresponding to multiple threshold provided.
        Keyword arguments:
        losses -- list of losses which the AUC should be calculated for
        thresholds -- list of threshold values up to which the AUC should be calculated
        binsize -- bin size to be used fo the cumulative histogram when calculating the AUC, the finer the more accurate
    """

    bin_num = int(max(thresholds) / binsize)
    bins = torch.arange(bin_num + 1) * binsize

    hist, _ = torch.histogram(losses, bins)  # histogram up to the max threshold
    hist = hist.astype(torch.float32) / len(losses)  # normalized histogram
    hist = torch.cumsum(hist)  # cumulative normalized histogram

    # calculate AUC for each threshold
    return [torch.mean(hist[:int(t / binsize)]) for t in thresholds]


# for checking, kornia
def cross_product_matrix(x):
    r"""Return the cross_product_matrix symmetric matrix of a vector.

    Args:
        x: The input vector to construct the matrix in the shape :math:`(*, 3)`.

    Returns:
        The constructed cross_product_matrix symmetric matrix with shape :math:`(*, 3, 3)`.

    """
    if not x.shape[-1] == 3:
        raise AssertionError(x.shape)
    # get vector components
    x0 = x[..., 0]
    x1 = x[..., 1]
    x2 = x[..., 2]

    # construct the matrix, reshape to 3x3 and return
    zeros = torch.zeros_like(x0)
    cross_product_matrix_flat = torch.stack([zeros, -x2, x1, x2, zeros, -x0, -x1, x0, zeros], dim=-1)
    shape_ = x.shape[:-1] + (3, 3)
    return cross_product_matrix_flat.view(*shape_)


def motion_from_essential_choose_solution(
    E_mat: torch.Tensor,
    K1: torch.Tensor,
    K2: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor) :
    """from kornia"""
    r"""Recover the relative camera rotation and the translation from an estimated essential matrix.

    The method checks the corresponding points in two images and also returns the triangulated
    3d points. Internally uses :py:meth:`~kornia.geometry.epipolar.decompose_essential_matrix` and then chooses
    the best solution based on the combination that gives more 3d points in front of the camera plane from
    :py:meth:`~kornia.geometry.epipolar.triangulate_points`.

    Args:
        E_mat: The essential matrix in the form of :math:`(*, 3, 3)`.
        K1: The camera matrix from first camera with shape :math:`(*, 3, 3)`.
        K2: The camera matrix from second camera with shape :math:`(*, 3, 3)`.
        x1: The set of points seen from the first camera frame in the camera plane
          coordinates with shape :math:`(*, N, 2)`.
        x2: The set of points seen from the first camera frame in the camera plane
          coordinates with shape :math:`(*, N, 2)`.
        mask: A boolean mask which can be used to exclude some points from choosing
          the best solution. This is useful for using this function with sets of points of
          different cardinality (for instance after filtering with RANSAC) while keeping batch
          semantics. Mask is of shape :math:`(*, N)`.

    Returns:
        The rotation and translation plus the 3d triangulated points.
        The tuple is as following :math:`[(*, 3, 3), (*, 3, 1), (*, N, 3)]`.

    """
    if not (len(E_mat.shape) >= 2 and E_mat.shape[-2:] == (3, 3)):
        raise AssertionError(E_mat.shape)
    if not (len(K1.shape) >= 2 and K1.shape[-2:] == (3, 3)):
        raise AssertionError(K1.shape)
    if not (len(K2.shape) >= 2 and K2.shape[-2:] == (3, 3)):
        raise AssertionError(K2.shape)
    if not (len(x1.shape) >= 2 and x1.shape[-1] == 2):
        raise AssertionError(x1.shape)
    if not (len(x2.shape) >= 2 and x2.shape[-1] == 2):
        raise AssertionError(x2.shape)
    if not len(E_mat.shape[:-2]) == len(K1.shape[:-2]) == len(K2.shape[:-2]):
        raise AssertionError
    # if mask is not None:
    #     if len(mask.shape) < 1:
    #         raise AssertionError(mask.shape)
    #     if mask.shape != x1.shape[:-1]:
    #         raise AssertionError(mask.shape)

    unbatched = len(E_mat.shape) == 2

    if unbatched:
        # add a leading batch dimension. We will remove it at the end, before
        # returning the results
        E_mat = E_mat[None]
        K1 = K1[None]
        K2 = K2[None]
        x1 = x1[None]
        x2 = x2[None]
        # if mask is not None:
        #     mask = mask[None]
    #print("check the A matrix in svd", E_mat, torch.isnan(E_mat).any(), [torch.matrix_rank(e) for e in E_mat])
    # compute four possible pose solutions
    Rs, ts = motion_from_essential(E_mat)

    # set reference view pose and compute projection matrix
    R1 = eye_like(3, E_mat)  # Bx3x3
    t1 = vec_like(3, E_mat)  # Bx3x1

    # compute the projection matrices for first camera
    R1 = R1[:, None].expand(-1, 4, -1, -1)
    t1 = t1[:, None].expand(-1, 4, -1, -1)
    K1 = K1[:, None].expand(-1, 4, -1, -1)
    P1 = projection_from_KRt(K1, R1, t1)  # 1x4x4x4

    # compute the projection matrices for second camera
    R2 = Rs
    t2 = ts
    K2 = K2[:, None].expand(-1, 4, -1, -1)
    P2 = projection_from_KRt(K2, R2, t2)  # Bx4x4x4

    # triangulate the points
    x1 = x1[:, None].expand(-1, 4, -1, -1)
    x2 = x2[:, None].expand(-1, 4, -1, -1)
    X = triangulate_points(P1, P2, x1, x2)  # Bx4xNx3

    # project points and compute their depth values
    d1 = depth_from_point(R1, t1, X)
    d2 = depth_from_point(R2, t2, X)

    # verify the point values that have a positive depth value
    depth_mask = (d1 > 0.0) & (d2 > 0.0)
    # if mask is not None:
    #     depth_mask &= mask.unsqueeze(1)

    mask_indices = torch.max(depth_mask.sum(-1), dim=-1, keepdim=True)[1]

    # get pose and points 3d and return
    R_out = Rs[:, mask_indices][:, 0, 0]
    t_out = ts[:, mask_indices][:, 0, 0]
    #points3d_out = X[:, mask_indices][:, 0, 0]

    if unbatched:
        R_out = R_out[0]
        t_out = t_out[0]
        #points3d_out = points3d_out[0]

    return R_out, t_out#, points3d_out


def f_error(pts1, pts2, F, gt_F, threshold):
    """ from NG-RANSAC Compute multiple evaluaton measures for a fundamental matrix.

    Return (False, 0, 0, 0) if the evaluation fails due to not finding inliers for the ground truth model,
    else return() True, F1 score, % inliers, mean epipolar error of inliers)
    Follows the evaluation procedure in:
    "Deep Fundamental Matrix Estimation"
    Ranftl and Koltun
    ECCV 201
    Keyword arguments:
    pts1 -- 3D numpy array containing the feature coordinates in image 1, dim 1: x and y coordinate, dim 2: correspondences, dim 3: dummy dimension
    pts2 -- 3D numpy array containing the feature coordinates in image 2, dim 1: x and y coordinate, dim 2: correspondences, dim 3: dummy dimension
    F -- 2D numpy array containing an estimated fundamental matrix
    gt_F -- 2D numpy array containing the corresponding ground truth fundamental matrix
    threshold -- inlier threshold for the epipolar error in pixels
    """

    EPS = 0.00000000001
    num_pts = pts1.shape[1]

    # 2D coordinates to 3D homogeneous coordinates
    hom_pts1 = np.concatenate((pts1[:, :, 0], np.ones((1, num_pts))), axis=0)
    hom_pts2 = np.concatenate((pts2[:, :, 0], np.ones((1, num_pts))), axis=0)

    def epipolar_error(hom_pts1, hom_pts2, F):
        """Compute the symmetric epipolar error."""
        res = 1 / np.linalg.norm(F.T.dot(hom_pts2)[0:2], axis=0)
        res += 1 / np.linalg.norm(F.dot(hom_pts1)[0:2], axis=0)
        res *= abs(np.sum(hom_pts2 * np.matmul(F, hom_pts1), axis=0))
        return res

    # determine inliers based on the epipolar error
    est_res = epipolar_error(hom_pts1, hom_pts2, F)
    gt_res = epipolar_error(hom_pts1, hom_pts2, gt_F)
    est_inliers = (est_res < threshold)
    gt_inliers = (gt_res < threshold)

    true_positives = est_inliers & gt_inliers
    gt_inliers = float(gt_inliers.sum())

    if gt_inliers > 0:
        est_inliers = float(est_inliers.sum())
        true_positives = float(true_positives.sum())
        precision = true_positives / (est_inliers + EPS)
        recall = true_positives / (gt_inliers + EPS)
        F1 = 2 * precision * recall / (precision + recall + EPS)
        inliers = est_inliers / num_pts
        epi_mask = (gt_res < 1)
        if epi_mask.sum() > 0:
            epi_error = float(est_res[epi_mask].mean())
        else:
            # no ground truth inliers for the fixed 1px threshold used for epipolar errors
            return False, 0, 0, 0
        return True, F1, inliers, epi_error
    else:
        # no ground truth inliers for the user provided threshold
        return False, 0, 0, 0


def pose_error(R, gt_R, t, gt_t):
    """NG-RANSAC, Compute the angular error between two rotation matrices and two translation vectors.

	Keyword arguments:
	R -- 2D numpy array containing an estimated rotation
	gt_R -- 2D numpy array containing the corresponding ground truth rotation
	t -- 2D numpy array containing an estimated translation as column
	gt_t -- 2D numpy array containing the corresponding ground truth translation
	"""

    # calculate angle between provided rotations
    dR = np.matmul(R, np.transpose(gt_R))
    dR = cv2.Rodrigues(dR)[0]
    dR = np.linalg.norm(dR) * 180 / math.pi

    # calculate angle between provided translations
    dT = float(np.dot(gt_t.T, t))
    dT /= float(np.linalg.norm(gt_t))

    if dT > 1 or dT < -1:
        print("Domain warning! dT:", dT)
        dT = max(-1, min(1, dT))
    dT = math.acos(dT) * 180 / math.pi

    return dR, dT


def batch_episym(x1, x2, F):
    """epipolar symmetric error from CLNet"""
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
    if torch.isnan(ys).any():
        print("ys is nan in batch_episym")
    return ys
