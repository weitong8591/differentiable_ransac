
from tqdm import tqdm
from model_cl import *
from loftr.loftr import LoFTR
from loftr.utils.cvpr_ds_config import default_cfg
from datasets import DatasetPicture
from tensorboardX import SummaryWriter
from loss import *
from ransac import RANSAC
from loftr.loftr_loss import LoFTRLoss
CUDA_LAUNCH_BLOCKING=2, 3
import logging
logger = logging.getLogger(__name__)

def train_step(train_data, opt, loss_fn, robust_estimator, topk_flag=False, valid_flag=False, k=1):
    for given_key in train_data.keys():
        try:
            train_data[given_key] = train_data[given_key].to(opt.device).to(torch.float32)
        except:
            train_data[given_key] = train_data[given_key]

    # fetch the points, ground truth extrinsic and intrinsic matrices
    pts1 = train_data['mkpts0_f'].to(opt.device).clone()
    pts2 = train_data['mkpts1_f'].to(opt.device).clone()
    K1, K2 = train_data['K1'].to(opt.device), train_data['K2'].to(opt.device)
    im_size1, im_size2 = torch.as_tensor(list(train_data['hw0_i'])).to(opt.device), torch.as_tensor(
        list(train_data['hw1_i'])).to(opt.device)
    pts1[:, 0] -= float(im_size1[1]) / 2
    pts1[:, 1] -= float(im_size1[0]) / 2
    pts1 /= float(max(im_size1))
    pts2[:, 0] -= float(im_size2[1]) / 2
    pts2[:, 1] -= float(im_size2[0]) / 2
    pts2 /= float(max(im_size2))

    gt_R, gt_t = train_data['gt_R'].to(opt.device), train_data['gt_t'].to(opt.device)
    gt_E = train_data['gt_E'].to(opt.device)
    gt_F = train_data['gt_F'].to(opt.device)
    points = torch.cat([pts1, pts2], dim=-1)
    confidence = train_data['mconf'].unsqueeze(dim=0)
    ground_truth = gt_F if opt.fmat else gt_E
    Es, ransac_time = robust_estimator.forward(
        points,
        confidence,
        K1,
        K2,
        im_size1,
        im_size2,
        ground_truth
    )
    pts1 = pts1.unsqueeze(0)
    pts2 = pts2.unsqueeze(0)
    im_size1 = im_size1.unsqueeze(0)
    im_size2 = im_size2.unsqueeze(0)
    loss = 0.
    for idx, f in enumerate(loss_fn):
        if opt.w[idx] != 0:
            if idx == 1:
                train_loss = 0

            elif idx == 2:
                ground_truth = ground_truth.to(torch.float32)
                pts1 = pts1.to(torch.float32)
                pts2 = pts2.to(torch.float32)
                K1 = K1.to(torch.float32)
                K2 = K2.to(torch.float32)
                train_loss = f.forward(
                    Es.unsqueeze(dim=0),
                    gt_E.cpu().detach().numpy().astype(np.float64),
                    pts1,
                    pts2,
                    K1,
                    K2,
                    im_size1,
                    im_size2,
                    topk_flag=topk_flag,
                    k=k
                )
            else:
                train_loss = f.forward(
                    Es.unsqueeze(0),
                    #gt_E,
                    pts1,
                    pts2,
                    gt_R,
                    gt_t,
                    K1,
                    K2,
                    im_size1,
                    im_size2,
                    topk_flag=topk_flag,
                    sssk=k,
                )
            loss += opt.w[idx] * train_loss
    return loss, Es, ransac_time


def train_one_epoch_loftr(
        model_loftr,
        optimizer_loftr,
        criterion_loftr,
        train_loader,
        valid_loader,
        opt,
        robust_estimator,
        epoch
    ):
    valid_loader_iter = iter(valid_loader)
    for idx, train_data in enumerate(tqdm(train_loader)):
        model_loftr.train()
        optimizer_loftr.zero_grad()

        # make sure all the data is on the same device.
        for given_key in train_data.keys():
            train_data[given_key] = train_data[given_key].to(opt.device)
        train_data['thr'] = 0.2
        try:
            model_loftr(train_data)
        except Exception as e:
            print('error in loftr FF: ', e, flush=True)
            continue
        if train_data['mkpts0_f'].shape[0] < 8:
            print('got too little samples in the fine!!!', flush=True)
            print(train_data['mkpts0_f'].shape, flush=True)
            continue

        loftr_loss, Es, run_time = train_step(
                train_data,
                opt,
                criterion_loftr,
                robust_estimator,
                topk_flag=opt.topk,
                k = opt.k,)

        if torch.isnan(loftr_loss):
            print('loss is nan', flush=True)
            continue

        loftr_loss.backward()
        torch.nn.utils.clip_grad_norm_(model_loftr.parameters(), max_norm=1.)
        optimizer_loftr.step()
        if torch.isnan(optimizer_loftr.param_groups[0]['params'][0]).any():
            continue
        else:
            print('train_loss: ', loftr_loss.item(), ' num_matches: ', train_data['mkpts0_f'].shape[-2], flush=True)
    return model_loftr, optimizer_loftr


if __name__ == '__main__':

    # Parse the parameters
    parser = create_parser(
        description="train the featuers matcher LoFTR with Generalized Differentiable RANSAC.")
    config = parser.parse_args()

    # check if gpu device is available
    config.device = torch.device('cuda:0' if torch.cuda.is_available() and config.device != 'cpu' else 'cpu')
    print(f"Running on {config.device}", flush=True)

    scenes = [config.datasets]

    model_loftr = LoFTR(default_cfg)
    model_loftr.load_state_dict(torch.load("pretrained_models/outdoor_ds.ckpt")['state_dict'])
    model_loftr = model_loftr.to(config.device)

    optimizer_loftr = torch.optim.AdamW(params=model_loftr.parameters(), lr=config.learning_rate)
    criterion_loftr = [
        PoseLoss(config.fmat),
        ClassificationLoss(config.fmat),
        MatchLoss(config.fmat)]

    diff_ransac = RANSACLayer(config)

    # normalize the weights for different losses
    w0, w1, w2 = config.w0, config.w1, config.w2,
    w_sum = w0 + w1 + w2

    if w_sum == 0:
        config.w = [1., 0., 0.]
    else:
        config.w = [float(w0 / w_sum), float(w1 / w_sum), float(w2 / w_sum)]
    scenes = config.datasets
    print(f'Working on {scenes} with scoring {config.scoring}')

    folders = config.data_path + '/' + scenes + '/' #seq + '/' for seq in scenes]
    train_dataset = DatasetPicture(folders, nfeatures=config.nfeatures, fmat=config.fmat)
    valid_dataset = DatasetPicture(folders, nfeatures=config.nfeatures, fmat=config.fmat, valid=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, num_workers=0,
                                               pin_memory=True, shuffle=True)
    print(f'Loading training data: {len(train_dataset)} image pairs.', flush=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.batch_size, num_workers=0,
                                               pin_memory=True, shuffle=True)
    print(f'Loading validation data: {len(valid_dataset)} image pairs.', flush=True)

    save_folder = create_session_string(
        'loftr_', config.sampler, config.epochs, config.fmat, config.nfeatures, config.snn, config.session, config.w0, config.w1, config.w2,
        config.threshold
    )
    src_path = 'results/loftr/' + save_folder
    if not os.path.isdir(src_path): os.makedirs(src_path)

    for given_epoch in range(config.epochs):  # train the LoFTR model
        model_loftr.train()

        model_loftr, optimizer_loftr = train_one_epoch_loftr(
            model_loftr,
            optimizer_loftr,
            criterion_loftr,
            train_loader,
            valid_loader,
            config,
            diff_ransac,
            given_epoch

        )

        print('Saving LoFTR model', flush=True)
        torch.save(model_loftr, src_path + '/loftr_model_' + str(given_epoch) + '.pth')
