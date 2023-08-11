import  os
import time
import torch
import numpy as np
import pymagsac
from tqdm import tqdm
from model_cl import *
from utils import *
from datasets import Dataset3D
from registration_utils import * 

def test(model, test_loader, opt):

    with torch.no_grad():
        avg_ransac_time = 0  # runtime of RANSAC

        # essential matrix evaluation
        pose_losses = []
        rtes    = []
        recalls =[]
        rmses = []
        for idx, test_data in enumerate(tqdm(test_loader)):

            correspondences, gt_pose = test_data['correspondences'].to(opt.device), test_data['gt_pose'].to(opt.device)

            batch_size = correspondences.size(0)

            # predicted inlier probabilities and normalization.
            inlier_weights = model(correspondences.transpose(-1, -2)[:, :, :, None])

            ransac_time = 0
            correspondences = correspondences.cpu().detach().numpy()[:, : ,  :6]
            for b in range(batch_size):

                if opt.use_conf:
                    weights = correspondences[b, :, -1]
                else:
                    weights = inlier_weights[b].cpu().detach().numpy()

                sorted_indices = np.argsort(weights)[::-1]

                # rank the points according to their probabilities
                sorted_pts = correspondences[b][sorted_indices]
                weights = (weights[sorted_indices]).astype(np.float64)
                start_time = time.time()
                pose, mask = pymagsac.findRigidTransformation(
                    np.ascontiguousarray(sorted_pts).astype(np.float64),
                        probabilities=weights,                        
                        use_magsac_plus_plus=True,
                        sigma_th=opt.threshold,
                        sampler=1,
                        max_iters = opt.max_iters
                    )
                
                ransac_time += time.time() - start_time
               
                # count inlier number
                incount = np.sum(mask)
                try:
                    pose = pose.T
                except:
                    pose = torch.eye(4).T
                rre, rte= compute_registration_error(gt_pose[b], pose)

                realignment_transform = torch.matmul(torch.inverse(gt_pose[b]), torch.from_numpy(pose).float())
                realigned_src_points_f = apply_transform(correspondences[b][:, :3], realignment_transform)
                rmse = torch.linalg.norm(realigned_src_points_f - correspondences[b][:, :3], dim=1).mean()
                recall = torch.lt(rmse, 0.2).float()
                pose_losses.append(rre)
                rtes.append(rte)
                recalls.append(recall)
                rmses.append(rmse)

            avg_ransac_time += ransac_time / batch_size

        out = 'results_rigid/' + (opt.model).replace('/', '_') + '/'
        print("RRE: %.2f RTE: %.2f RMSE: %.2f RR: %.2f " % (np.mean(pose_losses), np.mean(rtes)*100, np.mean(rmses)*100, np.mean(recalls)*100))        
        if not os.path.isdir(out): os.makedirs(out)
        with open(out + str(opt.max_iters) + '_' + str(opt.use_conf) + '_test.txt', 'a', 1) as f:
            f.write('%f %f %f %f %f ms '% (np.mean(pose_losses), np.mean(rtes), np.mean(rmses), np.mean(recalls), avg_ransac_time * 1000))
            f.write('\n') 


if __name__ == '__main__':

    # Parse the parameters
    parser = create_parser(
        description="Generalized Differentiable RANSAC, applied in point cloud registration.")
    parser.add_argument('--max_iters', '-max',type=int, default=1000,
                        help='maximal iterations for MAGSAC.')
    parser.add_argument('--use_conf', '-us',type=int, default=0,
                        help='sampling guided by the given matching confidence/our trained ones.')
    opt = parser.parse_args()
    opt.device = torch.device('cuda:0' if torch.cuda.is_available() and opt.device != 'cpu' else 'cpu')
    print(f"Running on {opt.device}")

    model = CLNet().to(opt.device)
    test_folders = [opt.data_path + '/' + i +'/' for i in os.listdir(opt.data_path)]

    dataset = Dataset3D(test_folders)
    test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=opt.batch_size, num_workers=0, pin_memory=False, shuffle=False)
    print(f'Loading test data: {len(dataset)} image pairs.')
    model.load_state_dict(torch.load(opt.model, map_location=opt.device))
    model.eval()
    test(model, test_loader, opt)


