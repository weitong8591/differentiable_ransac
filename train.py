import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from loss import *
from model_cl import *
from datasets import Dataset
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split


def train_step(train_data, model, opt, loss_fn):
    if opt.precision == 2:
        data_type = torch.float64
    elif opt.precision == 0:
        data_type = torch.float16
    else:
        data_type = torch.float32

    model.to(data_type)
    # fetch the points, ground truth extrinsic and intrinsic matrices
    correspondences, K1, K2 = train_data['correspondences'].to(opt.device, data_type), train_data['K1'].to(opt.device, data_type), \
                              train_data['K2'].to(opt.device, data_type)
    gt_R, gt_t = train_data['gt_R'].to(opt.device, data_type), train_data['gt_t'].to(opt.device, data_type)
    gt_E = train_data['gt_E'].to(opt.device, data_type)
    gt_F = train_data['gt_F'].to(opt.device, data_type)
    im_size1, im_size2 = train_data['im_size1'].to(opt.device, data_type), train_data['im_size2'].to(opt.device, data_type)

    ground_truth = gt_F if opt.fmat else gt_E
    if opt.tr:
        # default the best choice of weight types
        if opt.fmat and opt.sampler == 3:  # 8PC
            prob_type = 1
        elif opt.fmat and opt.sampler == 2:  # 7PC
            prob_type = 1
        else:  # 5PC
            prob_type = 0
    else:
        prob_type = opt.prob
    # collect all the models
    Es, weights, _ = model(
        correspondences.to(data_type),
        K1,
        K2,
        im_size1,
        im_size2,
        prob_type,
        ground_truth
    )

    pts1 = correspondences.squeeze(-1)[:, 0:2].transpose(-1, -2)
    pts2 = correspondences.squeeze(-1)[:, 2:4].transpose(-1, -2)

    loss = []
    for idx, f in enumerate(loss_fn):
        if opt.w[idx] != 0:
            if idx == 1:
                # w1, binary classification loss
                train_loss = f.forward(
                    ground_truth.cpu().detach().numpy(),
                    pts1,
                    pts2,
                    weights,
                    K1.cpu().detach().numpy(),
                    K2.cpu().detach().numpy(),
                    im_size1,
                    im_size2
                )

            elif idx == 2:
                train_loss = f.forward(
                    Es,
                    gt_E.cpu().detach().numpy(),
                    pts1,
                    pts2,
                    K1,
                    K2,
                    im_size1,
                    im_size2
                )
            else:
                # w0, pose error
                train_loss = f.forward_average(
                    Es,
                    pts1,
                    pts2,
                    gt_R,
                    gt_t,
                    K1,
                    K2,
                    im_size1,
                    im_size2
                )

            loss.append(opt.w[idx] * train_loss)

    return sum(loss), Es


def train(
        model,
        train_loader,
        valid_loader,
        opt
):
    # the name of the folder we save models, logs
    saved_file = create_session_string(
        "train",
        opt.sampler,
        opt.epochs,
        opt.fmat,
        opt.nfeatures,
        opt.snn,
        opt.session,
        opt.w0,
        opt.w1,
        opt.w2,
        opt.threshold
    )
    writer = SummaryWriter('results/' + saved_file + '/vision', comment="model_vis")
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    loss_function = [PoseLoss(opt.fmat), ClassificationLoss(opt.fmat), MatchLoss(opt.fmat)]
    valid_loader_iter = iter(valid_loader)

    # save the losses to npy file
    train_losses = []
    valid_losses = []

    # start epoch
    for epoch in range(opt.epochs):
        # each step
        for idx, train_data in enumerate(tqdm(train_loader)):
            # store the network every so often
            torch.save(model.state_dict(), 'results/' + saved_file + '/model.net')
            torch.save(model, 'results/' + saved_file + '/model_vision.pt')
            model.train()

            # one step
            optimizer.zero_grad()
            train_loss, Es = train_step(train_data, model, opt, loss_function)
            train_loss.retain_grad()
            for i in Es:
                i.retain_grad()
            # gradient calculation, ready for back propagation
            if torch.isnan(train_loss):
                print("pls check, there is nan value in loss!", train_loss)
                continue

            try:
                train_loss.backward()
                print("successfully back-propagation", train_loss)

            except Exception as e:
                print("we have trouble with back-propagation, pls check!", e)
                continue

            if torch.isnan(train_loss.grad):
                print("pls check, there is nan value in the gradient of loss!", train_loss.grad)
                continue

            for E in Es:
                if torch.isnan(E.grad).any():
                    print("pls check, there is nan value in the gradient of estimated models!", E.grad)
                    continue

            train_losses.append(train_loss.cpu().detach().numpy())
            # for vision
            writer.add_scalar('train_loss', train_loss, global_step=epoch*len(train_loader)+idx)

            # add gradient clipping after backward to avoid gradient exploding
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

            # check if the gradients of the training parameters contain nan values
            nans = sum([torch.isnan(param.grad).any() for param in list(model.parameters()) if param.grad is not None])
            if nans != 0:
                print("parameter gradients includes {} nan values".format(nans))
                continue

            optimizer.step()
            # check check if the training parameters contain nan values
            nan_num = sum([torch.isnan(param).any() for param in optimizer.param_groups[0]['params']])
            if nan_num != 0:
                print("parameters includes {} nan values".format(nan_num))
                continue

            print("_______________________________________________________")

            # validation
            with torch.no_grad():
                model.eval()
                try:
                    valid_data = next(valid_loader_iter)
                except StopIteration:
                    pass
                valid_loss, _ = train_step(valid_data, model, opt, loss_function)
                valid_losses.append(valid_loss.item())
                writer.add_scalar('valid_loss', valid_loss, global_step=epoch * len(train_loader) + idx)
            writer.flush()
            print('Step: {:02d}| Train loss: {:.4f}| Validation loss: {:.4f}'.format(
                epoch*len(train_loader)+idx,
                train_loss,
                valid_loss
            ), '\n')

    np.save('results/' + saved_file + '/' + 'loss_record.npy', (train_losses, valid_losses))


if __name__ == '__main__':

    OUT_DIR = 'results/'
    # Parse the parameters
    parser = create_parser(
        description="Fully Differentiable RANSAC.")
    config = parser.parse_args()
    # check if gpu device is available
    config.device = torch.device('cuda:0' if torch.cuda.is_available() and config.device != 'cpu' else 'cpu')
    print(f"Running on {config.device}")

    train_model = DeepRansac_CLNet(config).to(config.device)

    # use the pretrained model to initialize the weights if provided.
    if len(config.model) > 0:
        train_model.load_state_dict(torch.load(config.model))
    else:
        train_model.apply(init_weights)
    train_model.train()

    # normalize the weights for different losses
    w0, w1, w2 = config.w0, config.w1, config.w2
    w_sum = w0+w1+w2
    if w_sum == 0:
        # default loss, epipolar error if no weights are given.
        config.w = [0., 0., 1.]
    else:
        config.w = [float(w0 / w_sum), float(w1 / w_sum), float(w2 / w_sum)]

    # collect dataset list to be used for testing
    if config.batch_mode:
        scenes = test_datasets
        print("\n=== BATCH MODE: Training on", len(scenes), "datasets. =================")
    else:
        scenes = [config.datasets]
        print(f'Working on {scenes} with scoring {config.scoring}')

    folders = [config.data_path + '/' + seq + '/train_data_rs/' for seq in scenes]
    dataset = Dataset(
        folders,
        nfeatures=config.nfeatures,
        fmat=config.fmat
    )
    # split the data to train and validation
    train_dataset, valid_dataset = train_test_split(
        dataset,
        test_size=0.25,
        shuffle=True)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=True
    )
    print(f'Loading training data: {len(train_dataset)} image pairs.')
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=True
    )
    print(f'Loading validation data: {len(valid_dataset)} image pairs.')

    #with torch.autograd.set_detect_anomaly(True):
    train(train_model, train_data_loader, valid_data_loader, config)
