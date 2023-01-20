import math
import os
import cv2
import h5py
import numpy as np
import torch
import torch.utils.data as data
from loftr.util import load_torch_image, readh5, loadh5
import kornia as K


class Dataset(data.Dataset):
    """
    from NG-RANSAC
    collect the correspondences
    """
    def __init__(self, folders, ratiothreshold=0.8, nfeatures=2000, fmat=False):

        # access the input points

        self.nfeatures = nfeatures
        self.ratiothreshold = ratiothreshold
        self.fmat = fmat  # estimate fundamental matrix instead of essential matrix
        self.minset = 7 if self.fmat else 5
        self.hmat=0
        self.files = []

        for folder in folders:
             self.files += [folder + f for f in os.listdir(folder)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        data = np.load(self.files[index], allow_pickle=True, encoding='latin1')

        # correspondence coordinates and matching ratios (side information)
        pts1, pts2, ratios = data[0], data[1], data[2]
        # image sizes
        im_size1, im_size2 = torch.from_numpy(np.asarray(data[3])), torch.from_numpy(np.asarray(data[4]))
        # image calibration parameters
        K1, K2 = torch.from_numpy(data[5]), torch.from_numpy(data[6])
        # ground truth pose
        gt_R, gt_t = torch.from_numpy(data[7]), torch.from_numpy(data[8])
        # feature scale and orientation
        f_size1, f_size2 = torch.from_numpy(np.asarray(data[9])), torch.from_numpy(np.asarray(data[11]))
        ang1, ang2 = torch.from_numpy(np.asarray(data[10])), torch.from_numpy(np.asarray(data[12]))
        # des1, des2 = torch.from_numpy(data[13]), torch.from_numpy(data[14])

        # applying Lowes ratio criterion
        ratio_filter = ratios[0, :, 0] < self.ratiothreshold

        if ratio_filter.sum() < self.minset:  # ensure a minimum count of correspondences
            print("WARNING! Ratio filter too strict. Only %d correspondences would be left, so I skip it." % int(
                ratio_filter.sum()))
        else:
            pts1 = pts1[:, ratio_filter, :]
            pts2 = pts2[:, ratio_filter, :]
            ratios = ratios[:, ratio_filter, :]
            f_size1 = f_size1[:, ratio_filter, :]
            f_size2 = f_size2[:, ratio_filter, :]
            ang1 = ang1[:, ratio_filter, :]
            ang2 = ang2[:, ratio_filter, :]

        scale_ratio = f_size2 / f_size1
        ang = ((ang2 - ang1) % 180) * (3.141592653 / 180)

        if self.fmat or self.hmat:
            # for fundamental matrices, normalize image coordinates using the image size
            # (network should be independent to resolution)

            pts1[0, :, 0] -= float(im_size1[1]) / 2
            pts1[0, :, 1] -= float(im_size1[0]) / 2
            pts1 /= float(max(im_size1))
            pts2[0, :, 0] -= float(im_size2[1]) / 2
            pts2[0, :, 1] -= float(im_size2[0]) / 2
            pts2 /= float(max(im_size2))
            #utils.normalize_pts(pts1, im_size1)
            #utils.normalize_pts(pts2, im_size2)
            correspondences = np.concatenate((pts1, pts2, ratios, scale_ratio, ang), axis=2)
        else:
            # for essential matrices, normalize image coordinate using the calibration parameters

            pts1 = cv2.undistortPoints(pts1, K1.numpy(), None)
            pts2 = cv2.undistortPoints(pts2, K2.numpy(), None)
            # due to the opencv version issue, here transform it

            pts1_tran = list([j.tolist() for i in pts1 for j in i])
            pts2_tran = list([j.tolist() for i in pts2 for j in i])

        # stack image coordinates and side information into one tensor
            correspondences = np.concatenate((np.array([pts1_tran]), np.array([pts2_tran]), ratios, scale_ratio, ang),
                                             axis=2)
        # correspondences = np.concatenate((pts1, pts2, ratios, scale_ratio, ang), axis=2)
        correspondences = np.transpose(correspondences)
        correspondences = torch.from_numpy(correspondences)

        if self.nfeatures > 0:
            # ensure that there are exactly nfeatures entries in the data tensor
            if correspondences.size(1) > self.nfeatures:
                rnd = torch.randperm(correspondences.size(1))
                correspondences = correspondences[:, rnd, :]
                correspondences = correspondences[:, 0:self.nfeatures]

            if correspondences.size(1) < self.nfeatures:
                result = correspondences
                for i in range(0, math.ceil(self.nfeatures / correspondences.size(1) - 1)):
                    rnd = torch.randperm(correspondences.size(1))
                    result = torch.cat((result, correspondences[:, rnd, :]), dim=1)
                correspondences = result[:, 0:self.nfeatures]

        # construct the ground truth essential matrix from the ground truth relative pose
        gt_E = torch.zeros((3, 3), dtype=torch.float32)
        gt_E[0, 1] = -float(gt_t[2, 0])
        gt_E[0, 2] = float(gt_t[1, 0])
        gt_E[1, 0] = float(gt_t[2, 0])
        gt_E[1, 2] = -float(gt_t[0, 0])
        gt_E[2, 0] = -float(gt_t[1, 0])
        gt_E[2, 1] = float(gt_t[0, 0])

        gt_E = gt_E.mm(gt_R)

        # fundamental matrix from essential matrix
        gt_F = K2.inverse().transpose(0, 1).mm(gt_E).mm(K1.inverse())

        return {'correspondences': correspondences.float(), 'gt_F': gt_F, 'gt_E': gt_E, 'gt_R': gt_R, 'gt_t': gt_t,
                'K1': K1, 'K2': K2, 'im_size1': im_size1, 'im_size2': im_size2, 'files': self.files[index]}


class DatasetZero(data.Dataset):
    """
    from NG-RANSAC
    collect the correspondences
    """
    def __init__(self, folder, ratiothreshold=0.8, nfeatures=2000, fmat=False):

        # access the input points

        self.nfeatures = nfeatures
        self.ratiothreshold = ratiothreshold
        self.fmat = fmat  # estimate fundamental matrix instead of essential matrix
        self.minset = 7 if self.fmat else 5
        self.files = []

        self.files += [folder + f for f in os.listdir(folder)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        data = np.load(self.files[index], allow_pickle=True, encoding='latin1')

        # correspondence coordinates and matching ratios (side information)
        pts1, pts2, ratios = data[0], data[1], data[2]
        # image sizes
        im_size1, im_size2 = torch.from_numpy(np.asarray(data[3])), torch.from_numpy(np.asarray(data[4]))
        # image calibration parameters
        K1, K2 = torch.from_numpy(data[5]), torch.from_numpy(data[6])
        # ground truth pose
        gt_R, gt_t = torch.from_numpy(data[7]), torch.from_numpy(data[8])
        # feature scale and orientation
        f_size1, f_size2 = torch.from_numpy(np.asarray(data[9])), torch.from_numpy(np.asarray(data[11]))
        ang1, ang2 = torch.from_numpy(np.asarray(data[10])), torch.from_numpy(np.asarray(data[12]))
        # des1, des2 = torch.from_numpy(data[13]), torch.from_numpy(data[14])

        # applying Lowes ratio criterion
        ratio_filter = ratios[0, :, 0] < self.ratiothreshold

        if ratio_filter.sum() < self.minset:  # ensure a minimum count of correspondences
            print("WARNING! Ratio filter too strict. Only %d correspondences would be left, so I skip it." % int(
                ratio_filter.sum()))
        else:
            pts1 = pts1[:, ratio_filter, :]
            pts2 = pts2[:, ratio_filter, :]
            ratios = ratios[:, ratio_filter, :]
            f_size1 = f_size1[:, ratio_filter, :]
            f_size2 = f_size2[:, ratio_filter, :]
            ang1 = ang1[:, ratio_filter, :]
            ang2 = ang2[:, ratio_filter, :]

        scale_ratio = f_size2 / f_size1
        ang = ((ang2 - ang1) % 180) * (3.141592653 / 180)

        if self.fmat:
            # for fundamental matrices, normalize image coordinates using the image size
            # (network should be independent to resolution)

            pts1[0, :, 0] -= float(im_size1[1]) / 2
            pts1[0, :, 1] -= float(im_size1[0]) / 2
            pts1 /= float(max(im_size1))
            pts2[0, :, 0] -= float(im_size2[1]) / 2
            pts2[0, :, 1] -= float(im_size2[0]) / 2
            pts2 /= float(max(im_size2))
            #utils.normalize_pts(pts1, im_size1)
            #utils.normalize_pts(pts2, im_size2)
            correspondences = np.concatenate((pts1, pts2, ratios, scale_ratio, ang), axis=2)
        else:
            # for essential matrices, normalize image coordinate using the calibration parameters

            pts1 = cv2.undistortPoints(pts1, K1.numpy(), None)
            pts2 = cv2.undistortPoints(pts2, K2.numpy(), None)
            # due to the opencv version issue, here transform it

            pts1_tran = list([j.tolist() for i in pts1 for j in i])
            pts2_tran = list([j.tolist() for i in pts2 for j in i])

        # stack image coordinates and side information into one tensor
            correspondences = np.concatenate((np.array([pts1_tran]), np.array([pts2_tran]), ratios, scale_ratio, ang),
                                             axis=2)
        # correspondences = np.concatenate((pts1, pts2, ratios, scale_ratio, ang), axis=2)
        correspondences = np.transpose(correspondences)
        correspondences = torch.from_numpy(correspondences)

        if self.nfeatures > 0:
            # ensure that there are exactly nfeatures entries in the data tensor
            if correspondences.size(1) > self.nfeatures:
                rnd = torch.randperm(correspondences.size(1))
                correspondences = correspondences[:, rnd, :]
                correspondences = correspondences[:, 0:self.nfeatures]

            if correspondences.size(1) < self.nfeatures:
                result = correspondences
                correspondences = torch.zeros(correspondences.size(0), self.nfeatures, correspondences.size(2))
                correspondences[:, 0:result.size(1)] = result
        # construct the ground truth essential matrix from the ground truth relative pose
        gt_E = torch.zeros((3, 3), dtype=torch.float32)
        gt_E[0, 1] = -float(gt_t[2, 0])
        gt_E[0, 2] = float(gt_t[1, 0])
        gt_E[1, 0] = float(gt_t[2, 0])
        gt_E[1, 2] = -float(gt_t[0, 0])
        gt_E[2, 0] = -float(gt_t[1, 0])
        gt_E[2, 1] = float(gt_t[0, 0])

        gt_E = gt_E.mm(gt_R)

        # fundamental matrix from essential matrix
        gt_F = K2.inverse().transpose(0, 1).mm(gt_E).mm(K1.inverse())

        return {'correspondences': correspondences.float(), 'gt_F': gt_F, 'gt_E': gt_E, 'gt_R': gt_R, 'gt_t': gt_t,
                'K1': K1, 'K2': K2, 'im_size1': im_size1, 'im_size2': im_size2, 'files': self.files[index]}


class DatasetPictureTest(data.Dataset):
    """
    rewrite data collector based on  NG-RANSAC
    collect the correspondences
    """
    def __init__(self, folder, ratiothreshold=0.8, nfeatures=2000, fmat=False):

        # access the input points

        self.nfeatures = nfeatures
        self.ratiothreshold = ratiothreshold
        self.fmat = fmat  # estimate fundamental matrix instead of essential matrix
        self.minset = 7 if self.fmat else 5
        # with h5py.File(folder + 'Fgt.h5', 'r') as h5file:
        #     keys = list(h5file.keys())
        scene = folder.split('/')[-2]
        keys = np.load(folder.replace(scene + '/', 'evaluation_list/') + scene + '_list.npy')
        self.files = []

        self.files += [folder + f for f in os.listdir(folder)]
        self.files = sorted(self.files)
        self.files_dict = {}
        for given_file in self.files:
            if 'Egt.h5' in given_file:
                self.files_dict['gt_E'] = given_file
            elif 'Fgt.h5' in given_file:
                self.files_dict['gt_F'] = given_file
            elif 'K1_K2.h5' in given_file:
                self.files_dict['K1_K2'] = given_file
            elif 'R.h5' in given_file:
                self.files_dict['R'] = given_file
            elif 'T.h5' in given_file:
                self.files_dict['T'] = given_file
            elif '/images' in given_file:
                self.files_dict['img_dir'] = given_file
        self.pts1_list = []
        self.pts2_list = []
        for k in keys:
            #img_id1 = k.split('-')[0]
            #img_id2 = k.split('-')[1]
            img_id1 = k.split('_')[1] + '_' + k.split('_')[2]
            img_id2 = k.split('_')[3] + '_' + k.split('_')[4].split('.')[0]
            self.pts1_list.append(img_id1)
            self.pts2_list.append(img_id2)
        self.gt_F = loadh5(self.files_dict['gt_F'])
        self.gt_E =loadh5(self.files_dict['gt_E'])
        self.K1_K2 = loadh5(self.files_dict['K1_K2'])
        self.R = loadh5(self.files_dict['R'])
        self.T = loadh5(self.files_dict['T'])

    def __len__(self):
        return len(self.pts1_list)

    def __getitem__(self, index):
        img1 = load_torch_image(self.files_dict['img_dir'] + '/' + self.pts1_list[index] + '.jpg')
        img2 = load_torch_image(self.files_dict['img_dir'] + '/' + self.pts2_list[index] + '.jpg')
        match_id = self.pts1_list[index] + '-' + self.pts2_list[index]
        R1 = self.R[self.pts1_list[index]]
        R2 = self.R[self.pts2_list[index]]
        T1 = self.T[self.pts1_list[index]]
        T2 = self.T[self.pts2_list[index]]
        gt_R = np.dot(R2, R1.T)
        gt_t = T2 - np.dot(gt_R, T1)
        return {"image0": K.color.rgb_to_grayscale(img1).squeeze(0), # LofTR works on grayscale images only
                "image1": K.color.rgb_to_grayscale(img2).squeeze(0),
                'gt_F': torch.from_numpy(self.gt_F[match_id]),
                'gt_E': torch.from_numpy(self.gt_E[match_id]),
                'gt_R': gt_R, 'gt_t': gt_t,
                'K1': torch.from_numpy(self.K1_K2[match_id][0][0]),
                'K2': torch.from_numpy(self.K1_K2[match_id][0][1]),
                }


class DatasetPicture(data.Dataset):
    """
    rewrite data collector based on  NG-RANSAC
    collect the correspondences
    """
    def __init__(self, folder, ratiothreshold=0.8, nfeatures=2000, fmat=False):

        # access the input points

        self.nfeatures = nfeatures
        self.ratiothreshold = ratiothreshold
        self.fmat = fmat  # estimate fundamental matrix instead of essential matrix
        self.minset = 7 if self.fmat else 5
        with h5py.File(folder + 'Fgt.h5', 'r') as h5file:
            keys = list(h5file.keys())
        self.files = []

        self.files += [folder + f for f in os.listdir(folder)]
        self.files = sorted(self.files)
        self.files_dict = {}
        for given_file in self.files:
            if 'Egt.h5' in given_file:
                self.files_dict['gt_E'] = given_file
            elif 'Fgt.h5' in given_file:
                self.files_dict['gt_F'] = given_file
            elif 'K1_K2.h5' in given_file:
                self.files_dict['K1_K2'] = given_file
            elif 'R.h5' in given_file:
                self.files_dict['R'] = given_file
            elif 'T.h5' in given_file:
                self.files_dict['T'] = given_file
            elif '/images' in given_file:
                self.files_dict['img_dir'] = given_file
        self.pts1_list = []
        self.pts2_list = []
        for k in keys:
            img_id1 = k.split('-')[0]
            img_id2 = k.split('-')[1]
            self.pts1_list.append(img_id1)
            self.pts2_list.append(img_id2)
        self.gt_F = loadh5(self.files_dict['gt_F'])
        self.gt_E =loadh5(self.files_dict['gt_E'])
        self.K1_K2 = loadh5(self.files_dict['K1_K2'])
        self.R = loadh5(self.files_dict['R'])
        self.T = loadh5(self.files_dict['T'])

    def __len__(self):
        return len(self.pts1_list)

    def __getitem__(self, index):
        img1 = load_torch_image(self.files_dict['img_dir'] + '/' + self.pts1_list[index] + '.jpg')
        img2 = load_torch_image(self.files_dict['img_dir'] + '/' + self.pts2_list[index] + '.jpg')
        match_id = self.pts1_list[index] + '-' + self.pts2_list[index]
        R1 = self.R[self.pts1_list[index]]
        R2 = self.R[self.pts2_list[index]]
        T1 = self.T[self.pts1_list[index]]
        T2 = self.T[self.pts2_list[index]]
        gt_R = np.dot(R2, R1.T)
        gt_t = T2 - np.dot(gt_R, T1)
        return {"image0": K.color.rgb_to_grayscale(img1).squeeze(0), # LofTR works on grayscale images only
                "image1": K.color.rgb_to_grayscale(img2).squeeze(0),
                'gt_F': torch.from_numpy(self.gt_F[match_id]),
                'gt_E': torch.from_numpy(self.gt_E[match_id]),
                'gt_R': gt_R, 'gt_t': gt_t,
                'K1': torch.from_numpy(self.K1_K2[match_id][0][0]),
                'K2': torch.from_numpy(self.K1_K2[match_id][0][1]),
                }


