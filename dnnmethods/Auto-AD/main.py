# https://github.com/RSIDEA-WHU2020/Auto-AD
from __future__ import print_function
import matplotlib.pyplot as plt
# matplotlib inline

import os
import numpy as np
import time
import scipy.io
from models.skip import skip
import torch
import torch.optim
from utils.inpainting_utils import *
import shutil
import scipy.io as sio
import pdb
from sklearn.metrics import roc_auc_score, roc_curve

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
data_dir = '../../data/'
save_dir = '../../results/'

def main(file):
    # data input
    # **************************************************************************************************************
    print(file)
    data_path = data_dir + file + '.mat'
    save_subdir = os.path.join(save_dir, file)
    if not os.path.exists(save_subdir):
        os.makedirs(save_subdir)
    # load data
    mat = sio.loadmat(data_path)
    img_np = mat['data']
    img_np = img_np.transpose(2, 0, 1) # b, h, w
    # keep original simple normalization (no per-band changes)
    img_np = img_np - np.min(img_np)
    img_np = img_np / np.max(img_np) # [0, 1]
    gt = mat['map']
    img_var = torch.from_numpy(img_np).type(dtype)
    band, row, col = img_var.size()

    # training params
    thres = 0.00001
    channellss = 128
    layers = 5

    # model setup
    # **************************************************************************************************************
    pad = 'reflection' #'zero'
    OPT_OVER = 'net'
    method = '2D'
    input_depth = img_np.shape[0]
    LR = 1e-2
    num_iter = 1001
    param_noise = False
    reg_noise_std = 0.1 # 0 0.01 0.03 0.05

    # keep network architecture exactly as before
    net = skip(input_depth, img_np.shape[0],
               num_channels_down = [channellss] * layers,
               num_channels_up =   [channellss] * layers,
               num_channels_skip =    [channellss] * layers,
               filter_size_up = 3, filter_size_down = 3,
               upsample_mode='nearest', filter_skip_size=1,
               need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype).cuda()
    net = net.type(dtype)

    net_input = get_noise(input_depth, method, img_np.shape[1:]).type(dtype)
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    s  = sum(np.prod(list(p.size())) for p in net.parameters())
    print ('Number of params: %d' % s)

    # Loss
    mse = torch.nn.MSELoss().type(dtype)
    img_var = img_var[None, :].cuda()
    mask_var = torch.ones(1, band, row, col).cuda()
    residual_varr = torch.ones(row, col).cuda()

    # ---------- settings that affect only iterations / weighting ----------
    expected_anom_fraction = 0.10   # set to 0.05-0.10 depending on your dataset
    weight_floor = 0.01             # suppressed pixels get this weight
    early_iters = 500               # fast update phase length
    early_update_every = 10         # update mask every N iters during early phase
    later_update_every = 100        # then update every N iters after early phase

    def closure(iter_num, mask_varr, residual_varr):

        if param_noise:
            for n in [x for x in net.parameters() if len(x.size()) == 4]:
                n.data.add_(n.detach().clone().normal_() * n.std() / 50)

        net_input_local = net_input_saved
        if reg_noise_std > 0:
            net_input_local = net_input_saved + (noise.normal_() * reg_noise_std)

        out = net(net_input_local)
        out_np = out.detach().cpu().squeeze().numpy()

        mask_var_clone = mask_varr.detach().clone()
        residual_var_clone = residual_varr.detach().clone()

        # choose update frequency (fast early updates to prevent AE learning large anomalies)
        update_every = early_update_every if iter_num < early_iters else later_update_every

        if iter_num % update_every == 0 and iter_num != 0:
            # compute pixel-wise L2 residual across channels: sqrt(sum_c (out_c - img_c)^2)
            img_var_clone = img_var.detach().clone()
            net_output_clone = out.detach().clone()
            temp = (net_output_clone[0, :] - img_var_clone[0, :]) ** 2   # (B, H, W)
            residual_img = torch.sqrt(temp.sum(0) + 1e-12)              # (H, W)

            residual_var_clone = residual_img

            # percentile-based suppression: suppress top p fraction (likely anomalies)
            flat = residual_img.view(-1)
            n_pixels = flat.numel()
            k = int((1.0 - expected_anom_fraction) * n_pixels)
            if k < 1:
                k = 1
            # kthvalue returns the k-th smallest => quantile threshold P
            P = torch.kthvalue(flat, k).values.item()

            # build weights: 1 for residual <= P, weight_floor for residual > P
            weights = torch.where(residual_img <= P,
                                  torch.ones_like(residual_img),
                                  torch.ones_like(residual_img) * weight_floor)

            # broadcast same pixel weight across all channels
            # mask_var_clone shape: (1, band, H, W)
            mask_var_clone = mask_var_clone.clone()
            mask_var_clone[0, :, :, :] = weights.unsqueeze(0).expand(band, -1, -1)

        total_loss = mse(out * mask_var_clone, img_var * mask_var_clone)
        total_loss.backward()
        if iter_num % 50 == 0 or iter_num < 20:
            print("iteration: %d; loss: %f" % (iter_num+1, total_loss.item()))

        return mask_var_clone, residual_var_clone, out_np, total_loss

    net_input_saved = net_input.detach().clone()
    loss_np = np.zeros((1, 50), dtype=np.float32)
    loss_last = 0
    end_iter = False
    p = get_params(OPT_OVER, net, net_input)
    print('Starting optimization with ADAM')
    optimizer = torch.optim.Adam(p, lr=LR)

    start = time.time()
    for j in range(num_iter):
        optimizer.zero_grad()
        mask_var, residual_varr, background_img, loss = closure(j, mask_var, residual_varr)
        optimizer.step()

        if j >= 1:
            index = j-int(j/50)*50
            loss_np[0][index-1] = abs(loss-loss_last)
            if j % 50 == 0:
                mean_loss = np.mean(loss_np)
                if mean_loss < thres:
                    end_iter = True

        loss_last = loss

        if j == num_iter-1 or end_iter == True:
            residual_np = residual_varr.detach().cpu().squeeze().numpy()
            roc_auc = roc_auc_score(gt.flatten(), residual_np.flatten())
            print('Auc: %.4f' % roc_auc)
            # running time
            end = time.time()
            print("Runtime：%.2f" % (end - start))
            # save results
            fpr, tpr, thre = roc_curve(gt.flatten(), residual_np.flatten())
            map_path = os.path.join(save_subdir, "Auto-AD_map.mat")
            sio.savemat(map_path, {'show': residual_np})
            roc_path = os.path.join(save_subdir, "Auto-AD_roc.mat")
            sio.savemat(roc_path, {'PD': tpr, 'PF': fpr})

            return

if __name__ == "__main__":
    for file in ['porcine1_696x520x31']:
            #     ['los-angeles-1', 'los-angeles-2', 'gulfport', 
            # 'texas-goast', 'cat-island', 'pavia']:
        main(file)