import torch
from tqdm import tqdm
from model_cl import *
from utils import *
from datasets import Dataset


def test(model, test_loader, opt):

    OUT_DIR = "results/"

    with torch.no_grad():
        if opt.precision == 2:
            data_type = torch.float64
        elif opt.precision == 0:
            data_type = torch.float16
        else:
            data_type = torch.float32

        errRs, errTs = [], []
        max_errors = []
        avg_F1 = 0
        avg_inliers = 0
        epi_errors = []
        avg_ransac_time = 0
        invalid_pairs = 0
        model.to(data_type)

        for idx, test_data in enumerate(tqdm(test_loader)):

            correspondences, K1, K2 = test_data['correspondences'].to(opt.device, data_type), test_data['K1'].to(opt.device, data_type), \
                                      test_data['K2'].to(opt.device, data_type)
            im_size1, im_size2 = test_data['im_size1'].to(opt.device, data_type), test_data['im_size2'].to(opt.device, data_type)
            gt_F, gt_E, gt_R, gt_t = test_data['gt_F'].to(data_type), test_data['gt_E'].to(data_type), \
                                     test_data['gt_R'].to(opt.device, data_type), test_data['gt_t'].to(data_type)
            files = test_data['files']
            # estimate model, return the model, predicted inlier probablities and normalization.
            models, weights, ransac_time = model(correspondences, K1, K2, im_size1, im_size2)
            K1_, K2_ = K1.cpu().detach().numpy(), K2.cpu().detach().numpy()
            gt_F = gt_F.cpu().detach().numpy()
            im_size1, im_size2 = im_size1.cpu().detach().numpy(), im_size2.cpu().detach().numpy()

            for b, est_model in enumerate(models):
                if opt.fmat:
                    E = K2[b].transpose(-1, -2) @ est_model @ K1[b]
                    denormalize_pts_inplace(correspondences[b, 0:2], im_size1[b])
                    denormalize_pts_inplace(correspondences[b, 2:4], im_size2[b])

                    pts1 = correspondences[b, 0:2].cpu().detach().numpy()
                    pts2 = correspondences[b, 2:4].cpu().detach().numpy()
                    # evaluation of F matrix via correspondences
                    valid, F1, epi_inliers, epi_error = f_error(
                        pts1,
                        pts2,
                        est_model.cpu().detach().numpy(),
                        gt_F[b],
                        0.75
                    )

                    if valid:
                        avg_F1 += F1
                        avg_inliers += epi_inliers
                        epi_errors.append(epi_error)
                    else:
                        # F matrix evaluation failed (ground truth model had no inliers)
                        invalid_pairs += 1

                    # normalize the points for E error calculation
                    pts1 = cv2.undistortPoints(pts1.transpose(2, 1, 0), K1_[b], None)
                    pts2 = cv2.undistortPoints(pts2.transpose(2, 1, 0), K2_[b], None)
                    errR, errT = eval_essential_matrix(pts1.squeeze(), pts2.squeeze(), E, gt_R[b], gt_t[b])
                else:
                    pts1 = correspondences[b, 0:2].squeeze(-1).cpu().detach().numpy().T
                    pts2 = correspondences[b, 2:4].squeeze(-1).cpu().detach().numpy().T
                    E = est_model
                    errR, errT = eval_essential_matrix(pts1, pts2, E, gt_R[b], gt_t[b])

                errRs.append(float(errR))
                errTs.append(float(errT))
                max_errors.append(max(float(errR), float(errT)))
            avg_ransac_time += ransac_time

    avg_ransac_time /= len(test_loader)
    out = OUT_DIR + opt.model
    print(f"Rotation error = {np.mean(np.array(errRs))} | Translation error = {np.mean(np.array(errTs))}")
    print(f"Rotation error median= {np.median(np.array(errRs))} | Translation error median= {np.median(np.array(errTs))}")
    print(f"AUC scores = {AUC(max_errors)} ")
    if opt.fmat:
        avg_F1 /= len(epi_errors)
        avg_inliers /= len(epi_errors)
        epi_errors.sort()
        mean_epi_err = sum(epi_errors) / len(epi_errors)
        median_epi_err = epi_errors[int(len(epi_errors) / 2)]
        print("Invalid Pairs (ignored in the following metrics):", invalid_pairs)
        print("F1 Score: %.2f%%" % (avg_F1 * 100))
        print("%% Inliers: %.2f%%" % (avg_inliers * 100))
        print("Mean Epi Error: %.2f" % mean_epi_err)
        print("Median Epi Error: %.2f" % median_epi_err)

    print("Run time: %.2f ms" % (avg_ransac_time * 1000))
    if not os.path.isdir(out): os.makedirs(out)
    with open(out + '/test.txt', 'a', 1) as f:
        f.write('%f %f %f %f ms'% (AUC(max_errors)[0], AUC(max_errors)[1], AUC(max_errors)[2], avg_ransac_time * 1000))
        if opt.fmat and len(epi_errors) > 0:
            f.write('%f %f %f %f %f ms' % (avg_F1, avg_inliers, mean_epi_err, median_epi_err, avg_ransac_time * 1000))
        f.write('\n')


if __name__ == '__main__':
    # Parse the parameters
    parser = create_parser(
        description="Generalized Differentiable RANSAC.")
    opt = parser.parse_args()
    # check if gpu device is available
    opt.device = torch.device('cuda:0' if torch.cuda.is_available() and opt.device != 'cpu' else 'cpu')
    print(f"Running on {opt.device}")

    # collect datasets to be used for testing
    if opt.batch_mode:
        scenes = test_datasets
        print("\n=== BATCH MODE: Doing evaluation on", len(scenes), "datasets. =================")
    else:
        scenes = [opt.datasets]

    model = DeepRansac_CLNet(opt).to(opt.device)

    for seq in scenes:
        print(f'Working on {seq} with scoring {opt.scoring}')
        scene_data_path = os.path.join(opt.data_path)

        dataset = Dataset([scene_data_path + '/' + seq+'/test_data_rs/'],
                               opt.snn, nfeatures=opt.nfeatures, fmat=opt.fmat)

        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=opt.batch_size, num_workers=0, pin_memory=False, shuffle=False
        )
        print(f'Loading test data: {len(dataset)} image pairs.')

        # if opt.model is not None:
        model.load_state_dict(torch.load(opt.model, map_location=opt.device))
        model.eval()
        test(model, test_loader, opt)





