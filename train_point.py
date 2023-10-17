import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from model_cl import *
from datasets import Dataset3D
from tensorboardX import SummaryWriter

def train_step(train_data, weight_model, robust_estimator, data_type, prob_type=0, dev='cuda'):

    weight_model.to(data_type)
    # fetch the points, ground truth extrinsic and intrinsic matrices
    correspondences, gt_pose = train_data['correspondences'].to(dev, data_type), \
    train_data['gt_pose'].to(dev, data_type)
    # 1. importance score prediction
    weights = weight_model(correspondences.transpose(-1, -2)[:, :, :, None])
    # import pdb; pdb.set_trace()
    # 2. ransac
    loss_back = 0
    for i, pair in enumerate(correspondences[:, :, :6]):

        Es, loss, avg_loss, _ = robust_estimator(
            pair,
            weights[i],
            gt_pose[i]
        )

        loss_back += avg_loss

    return loss_back/correspondences.shape[0]


def train(
        model,
        estimator,
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
    writer = SummaryWriter('results/point/' + saved_file + '/vision', comment="model_vis")
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    valid_loader_iter = iter(valid_loader)

    # save the losses to npy file
    train_losses = []
    valid_losses = []

    if opt.precision == 2:
        data_type = torch.float64
    elif opt.precision == 0:
        data_type = torch.float16
    else:
        data_type = torch.float32

    # start epoch
    for epoch in range(opt.epochs):
        # each step
        for idx, train_data in enumerate(tqdm(train_loader)):

            model.train()
            # one step
            optimizer.zero_grad()
            train_loss = train_step(train_data, model, estimator, data_type, prob_type=opt.prob, dev=opt.device)
            train_loss.retain_grad()
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

        torch.save(model.state_dict(), 'results/point/' + saved_file + '/model' + str(epoch) + '.net')
        print("_______________________________________________________")

        # validation
        with torch.no_grad():
            model.eval()
            try:
                valid_data = next(valid_loader_iter)
            except StopIteration:
                pass

            valid_loss = train_step(valid_data, model, estimator, data_type, prob_type=opt.prob, dev=opt.device)
            valid_losses.append(valid_loss)
            writer.add_scalar('valid_loss', valid_loss, global_step=epoch * len(train_loader) + idx)
            writer.flush()
            print('Step: {:02d}| Train loss: {:.4f}| Validation loss: {:.4f}'.format(
                    epoch*len(train_loader)+idx,
                    train_loss,
                    valid_loss
                ), '\n')

    np.save('results/point/' + saved_file + '/' + 'loss_record.npy', (train_losses, valid_losses))


if __name__ == '__main__':

    # Parse the parameters
    parser = create_parser(
        description="Generalized Differentiable RANSAC.")

    config = parser.parse_args()

    # check if gpu device is available
    config.device = torch.device('cuda:0' if torch.cuda.is_available() and config.device != 'cpu' else 'cpu')
    print(f"Running on {config.device}")

    train_model = CLNet().to(config.device)
    robust_estimator = RANSACLayer3D(config)
    # use the pretrained model to initialize the weights if provided.
    if config.model is not None:
        train_model.load_state_dict(torch.load(config.model))
    else:
        train_model.apply(init_weights)
    train_model.train()

    # collect dataset list
    train_scenes = os.listdir(config.data_path)

    train_folders = [config.data_path + '/' + i  + '/' for i in train_scenes]
    train_dataset = Dataset3D(train_folders)
    v_folders = [config.data_path.replace('train', 'val')  + '/' + i + '/'  for i in os.listdir(config.data_path.replace('train', 'val'))]
    v_dataset = Dataset3D(v_folders)

    print("\n=== BATCH MODE: Training and validation on", len(train_scenes), len(v_folders), "datasets. =================")
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=True
    )
    print(f'Loading training data: {len(train_dataset)} image pairs.')
    valid_data_loader = torch.utils.data.DataLoader(
        v_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=True
    )

    print(f'Loading validation data: {len(v_dataset)} image pairs.')

    train(train_model, robust_estimator, train_data_loader, valid_data_loader, config)
