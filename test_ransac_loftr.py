from tqdm import tqdm
from datasets import DatasetPictureTest
from model_cl import *
from utils import *
from loftr.loftr import LoFTR
from loftr.utils.cvpr_ds_config import default_cfg


def test(model_loftr, test_loader, opt):
    with torch.no_grad():
        errRs, errTs = [], []
        max_errors = []
        avg_ransac_time = 0
        avg_loftr_time = 0
        avg_F1 = 0
        avg_inliers = 0
        epi_errors = []
        invalid_pairs = 0
        for idx, test_data in enumerate(tqdm(test_loader)):
            for given_key in test_data.keys():
                try:
                    test_data[given_key] = test_data[given_key].to(opt.device).to(torch.float32)
                except:
                    test_data[given_key] = test_data[given_key]

            model_loftr.to(opt.device)
            start_time = time.time()
            try:
                test_data['thr'] = 0.2
                model_loftr(test_data)
            except:
                try:
                    test_data['thr'] = 0.1
                    model_loftr(test_data)
                except:
                    try:
                        test_data['thr'] = 0.05
                        model_loftr(test_data)
                    except:
                        try:
                            test_data['thr'] = 0.02
                            model_loftr(test_data)
                        except Exception as e:
                            print('error in loftr FF: ', e, flush=True)
                            continue
            avg_loftr_time += time.time() - start_time
            if test_data['mkpts0_f'].shape[0] < 8:
                test_data['thr'] = 0.1
                model_loftr(test_data)
                if test_data['mkpts0_f'].shape[0] < 8:
                    test_data['thr'] = 0.05
                    model_loftr(test_data)
                    if test_data['mkpts0_f'].shape[0] < 8:
                        test_data['thr'] = 0.02
                        model_loftr(test_data)
                        if test_data['mkpts0_f'].shape[0] < 8:
                            print('got too little samples in the fine!!!', flush=True)
                            print(test_data['mkpts0_f'].shape, flush=True)
                            continue
            pts1 = test_data['mkpts0_f'].to(opt.device).clone()
            pts2 = test_data['mkpts1_f'].to(opt.device).clone()
            K1, K2 = test_data['K1'].to(opt.device), test_data['K2'].to(opt.device)

            gt_R, gt_t = test_data['gt_R'].to(opt.device), test_data['gt_t'].to(opt.device)
            gt_F = test_data['gt_F'].to(opt.device)
            confidence = test_data['mconf'].unsqueeze(dim=0)

            start_time = time.time()
            
            if opt.ransac == 0:
                F, _ = cv2.findFundamentalMat(
                    pts1.detach().cpu().numpy(), pts2.detach().cpu().numpy(),# threshold=1., prob=0.99999,
                    method=cv2.RANSAC)#np.eye(3),
            elif opt.ransac == 1:
                sorted_indices = np.argsort(confidence.detach().cpu().numpy())[::-1]
                sorted_pts1 = pts1.detach().cpu().numpy()[sorted_indices]
                sorted_pts2 = pts2.detach().cpu().numpy()[sorted_indices]
                F, _ = cv2.findFundamentalMat(
                    sorted_pts1, sorted_pts2,# np.eye(3), #threshold=1., prob=0.99999,
                    method=cv2.USAC_PROSAC)
            avg_ransac_time += time.time() - start_time

            if opt.fmat:

                try:
                    valid, F1, epi_inliers, epi_error = f_error(pts1.transpose(0, 1).unsqueeze(-1).cpu().detach().numpy(),
                                                            pts2.transpose(0, 1).unsqueeze(-1).cpu().detach().numpy(),
                                                            F,
                                                            gt_F[0].cpu().detach().numpy(), opt.threshold)
                except:
                    valid, F1, epi_inliers, epi_error = False, 0, 0, 0

                if valid:
                    avg_F1 += F1
                    avg_inliers += epi_inliers
                    epi_errors.append(epi_error)
                else:
                    invalid_pairs += 1
            else:
                pts1 = pts1.cpu().detach().numpy()
                pts2 = pts2.cpu().detach().numpy()
                    # normalize points for pose estimation
                E = K2[0].numpy().T.dot(F.dot(K1[0].numpy()))
                errR, errT = eval_essential_matrix(pts1, pts2, E, gt_R[0], gt_t[0])
                errRs.append(float(errR))
                errTs.append(float(errT))
                max_errors.append(max(float(errR), float(errT)))

    avg_ransac_time /= len(test_loader)
    if opt.fmat:
        avg_F1 /= len(epi_errors)
        avg_inliers /= len(epi_errors)
        epi_errors.sort()
        mean_epi_err = sum(epi_errors) / len(epi_errors)
        median_epi_err = epi_errors[int(len(epi_errors) / 2)]
        print("Invalid Pairs (ignored in the following metrics):", invalid_pairs, flush=True)
        print("F1 Score: %.2f%%" % (avg_F1 * 100), flush=True)
        print("%% Inliers: %.2f%%" % (avg_inliers * 100), flush=True)
        print("Mean Epi Error: %.2f" % mean_epi_err, flush=True)
        print("Median Epi Error: %.2f" % median_epi_err, flush=True)
    else:
        print(f"Rotation error = {np.mean(np.array(errRs))} | Translation error = {np.mean(np.array(errTs))}", flush=True)
        print(f"Rotation error median= {np.median(np.array(errRs))} | Translation error median= {np.median(np.array(errTs))}", flush=True)
        print(f"AUC scores = {AUC(max_errors)} ", flush=True)

    print("Run time: %.2fms" % (avg_ransac_time * 1000), flush=True)

    # write evaluation results to file
    remove_pth = opt.model_loftr.split('/')[-1]
    name = remove_pth.split('.')[0]
    save_pth = 'results/loftr/' + opt.model_loftr.replace(remove_pth, '')
    if not os.path.isdir(save_pth): os.makedirs(save_pth)
    with open(save_pth + name + '.txt', 'a', 1) as f:
        if opt.fmat and len(epi_errors) > 0:
            f.write(
                ' %f %f %f %f %fms' % (avg_F1, avg_inliers, mean_epi_err, median_epi_err, avg_ransac_time * 1000)
            )
        else:
            f.write('%f %f %f %fms %fms '% (AUC(max_errors)[0], AUC(max_errors)[1], AUC(max_errors)[2],
                                            avg_ransac_time * 1000, avg_loftr_time * 1000))

        f.write('\n')


if __name__ == '__main__':

    scenes = outdoor_test_datasets

    # Parse the parameters
    parser = create_parser(
        description="LoFTR + Generalized differentiable RANSAC.")
    parser.add_argument('--ransac', '-ransac',type=int, default=0,
                        help='0 OpenCV-RANSAC, 1 OpenCV-MAGSAC, 2-MAGSAC++ with PROSAC.')
    opt = parser.parse_args()

    print(f"Running on {opt.device}", flush=True)

    model_loftr = LoFTR(default_cfg).to(opt.device)

    if 'outdoor_ds.ckpt' in opt.model_loftr:
        model_loftr.load_state_dict(torch.load(opt.model_loftr)['state_dict'])# pretrained:
    else:
        model_loftr = torch.load(opt.model_loftr)

    for seq in scenes:
        print(f'Working on {seq} with scoring {opt.scoring}', flush=True)

        model_loftr.eval()

        scene_data_path = os.path.join(opt.data_path)

        dataset = DatasetPictureTest(scene_data_path + '/' + seq + '/',
                                 opt.snn, nfeatures=opt.nfeatures, fmat=opt.fmat)
        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, pin_memory=False, shuffle=False)
        print(f'Loading test data: {len(dataset)} image pairs.', flush=True)
        
        test(model_loftr, test_loader, opt)