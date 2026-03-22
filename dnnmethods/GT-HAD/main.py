# https://github.com/jeline0110/GT-HAD
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.optim as Optim
import scipy.io as sio
import pdb
from net import Net
from sklearn.metrics import roc_auc_score, roc_curve
import shutil
from utils import get_params, img2mask, seed_dict
import random
from progress.bar import Bar
import time 
import torch.nn as nn 
from torch.utils.data import DataLoader
from data import DatasetHsi
from block import Block_fold, Block_search
# import cv2 

DEFAULT_DATA_DIR = "../../data/"
DEFAULT_SAVE_DIR = "../../results/"
DEFAULT_CUDA_VISIBLE_DEVICES = "0"

# The original code uses CUDA tensors directly.
dtype = torch.cuda.FloatTensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="GT-HAD trainer/inferencer with tunable hyperparameters."
    )
    parser.add_argument("--file", type=str, required=True,
                        help="Dataset stem name (e.g. los-angeles-1) or .mat filename if you prefer.")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Folder containing input .mat files.")
    parser.add_argument("--save-dir", type=str, default=DEFAULT_SAVE_DIR,
                        help="Root folder for experiment outputs.")
    parser.add_argument("--cuda-visible-devices", type=str, default=DEFAULT_CUDA_VISIBLE_DEVICES,
                        help="CUDA_VISIBLE_DEVICES value to set before model creation.")

    # Geometry / model knobs exposed for search.
    parser.add_argument("--patch-size", type=int, default=3,
                        help="Patch size used inside the transformer block.")
    parser.add_argument("--patch-stride", type=int, default=3,
                        help="Patch stride used inside the transformer block.")
    parser.add_argument("--block-stride", type=int, default=3,
                        help="Sliding-window stride used by DatasetHsi / Block_fold / Block_search.")
    parser.add_argument("--embed-dim", type=int, default=64,
                        help="Channel width of the model.")
    parser.add_argument("--mlp-ratio", type=float, default=2.0,
                        help="MLP expansion ratio in the transformer block.")
    parser.add_argument("--attn-drop", type=float, default=0.0,
                        help="Attention dropout.")
    parser.add_argument("--drop", type=float, default=0.0,
                        help="MLP dropout.")

    # Training knobs.
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training / inference batch size.")
    parser.add_argument("--end-iter", type=int, default=150,
                        help="Total training epochs/iterations.")
    parser.add_argument("--search-iter", type=int, default=25,
                        help="CMM update interval (in epochs).")

    # Residual diffusion (paper uses 3x3x5; code order is bands x rows x cols).
    parser.add_argument("--rd-bands", type=int, default=5,
                        help="Temporal/spectral kernel size for AvgPool3d on the band axis.")
    parser.add_argument("--rd-rows", type=int, default=3,
                        help="AvgPool3d kernel size on the row axis.")
    parser.add_argument("--rd-cols", type=int, default=3,
                        help="AvgPool3d kernel size on the col axis.")

    parser.add_argument("--seed", type=int, default=None,
                        help="Optional manual seed override.")
    parser.add_argument("--report-json", type=str, default=None,
                        help="Optional path to write a one-file JSON summary.")
    parser.add_argument("--save-heatmap", action="store_true", default=True,
                        help="Save anomaly heatmap .mat file with variable 'show'.")
    parser.add_argument("--save-roc", action="store_true", default=True,
                        help="Save ROC curve .mat file.")
    return parser.parse_args()


def resolve_file_path(file_arg: str, data_dir: str) -> tuple[str, str]:
    if file_arg.endswith(".mat"):
        stem = Path(file_arg).stem
        path = Path(file_arg)
    else:
        stem = file_arg
        path = Path(data_dir) / f"{file_arg}.mat"
    return stem, str(path)


def run_one(file: str, args) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GT-HAD in this repo is CUDA-oriented. Please run on a machine with a working NVIDIA GPU."
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    stem, data_path = resolve_file_path(file, args.data_dir)
    save_subdir = Path(args.save_dir) / stem
    save_subdir.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    if seed is None:
        seed = seed_dict.get(stem, 0)
    set_seed(seed)

    print(stem)
    print(f"Data path: {data_path}")

    mat = sio.loadmat(data_path)
    img_np = mat["data"].transpose(2, 0, 1)  # b, h, w
    img_np = img_np - np.min(img_np)
    img_np = img_np / np.max(img_np)
    gt = mat["map"]

    img_var = torch.from_numpy(img_np).type(dtype)
    band, row, col = img_var.size()
    img_var = img_var[None, :]

    # Geometry / block setup
    patch_size = args.patch_size
    patch_stride = args.patch_stride
    block_size = patch_size * patch_stride  # sliding-window cube size
    data_set = DatasetHsi(img_var, wsize=block_size, wstride=args.block_stride)
    block_fold = Block_fold(wsize=block_size, wstride=args.block_stride)
    block_search = Block_search(img_var, wsize=block_size, wstride=args.block_stride)
    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    # Model
    net = Net(
        in_chans=band,
        embed_dim=args.embed_dim,
        patch_size=patch_size,
        patch_stride=patch_stride,
        mlp_ratio=args.mlp_ratio,
        attn_drop=args.attn_drop,
        drop=args.drop,
    ).cuda()

    n_params = sum(np.prod(list(p.size())) for p in net.parameters())
    print(f"Number of params: {n_params}")

    mse = torch.nn.MSELoss().type(dtype)

    LR = args.lr
    p = get_params(net)
    optimizer = torch.optim.Adam(p, lr=LR)
    print("Starting optimization with ADAM")

    end_iter = args.end_iter
    search_iter = args.search_iter
    bar = Bar("Processing", max=end_iter)

    data_num = len(data_set)
    match_vec = torch.zeros((data_num)).type(dtype)
    search_matrix = torch.zeros((data_num, band, block_size, block_size)).type(dtype)
    search_index = torch.arange(0, data_num).type(torch.cuda.LongTensor)
    avgpool = nn.AvgPool3d(
        kernel_size=(args.rd_bands, args.rd_rows, args.rd_cols),
        stride=(1, 1, 1),
        padding=(args.rd_bands // 2, args.rd_rows // 2, args.rd_cols // 2),
    )

    start = time.time()
    for iter_idx in range(1, end_iter + 1):
        search_flag = True if iter_idx % search_iter == 0 and iter_idx != end_iter else False

        net.train()
        for _, batch_data in enumerate(data_loader):
            optimizer.zero_grad()
            net_gt = batch_data["block_gt"]
            net_input = batch_data["block_input"]
            block_idx = batch_data["index"].cuda()

            net_out = net(net_input, block_idx=block_idx, match_vec=match_vec)
            if search_flag:
                search_matrix[block_idx] = net_out

            loss = mse(net_out, net_gt)
            loss.backward()
            optimizer.step()

        if search_flag:
            match_vec = torch.zeros((data_num)).type(dtype)
            search_back = block_fold(search_matrix.detach(), data_set.padding, row, col)
            match_vec = block_search(search_back.detach(), match_vec, search_index)

        bar.next()

        if iter_idx == end_iter:
            bar.finish()
            infer_loader = torch.utils.data.DataLoader(
                data_set, batch_size=args.batch_size, shuffle=False, drop_last=False
            )
            net = net.eval()
            infer_res_list = []

            with torch.no_grad():
                for _, data in enumerate(infer_loader):
                    infer_in = data["block_input"]
                    infer_idx = data["index"].cuda()
                    infer_out = net(infer_in, block_idx=infer_idx, match_vec=match_vec)
                    infer_res = torch.abs(infer_in - infer_out) ** 2
                    infer_res = avgpool(infer_res)
                    infer_res_list.append(infer_res)

            infer_res_out = torch.cat(infer_res_list, dim=0)
            infer_res_back = block_fold(infer_res_out.detach(), data_set.padding, row, col)
            residual_np = img2mask(infer_res_back)

            auc = roc_auc_score(gt.flatten(), residual_np.flatten())
            print(f"Auc: {auc:.4f}")

            fpr, tpr, thre = roc_curve(gt.flatten(), residual_np.flatten())
            if args.save_heatmap:
                sio.savemat(save_subdir / "GT-HAD_map.mat", {"show": residual_np})
            if args.save_roc:
                sio.savemat(save_subdir / "GT-HAD_roc.mat", {"PD": tpr, "PF": fpr})

            end = time.time()
            runtime_sec = end - start
            print(f"Runtime：{runtime_sec:.2f}")

            summary = {
                "file": stem,
                "data_path": data_path,
                "auc": float(auc),
                "runtime_sec": float(runtime_sec),
                "config": {
                    "patch_size": patch_size,
                    "patch_stride": patch_stride,
                    "block_stride": args.block_stride,
                    "embed_dim": args.embed_dim,
                    "mlp_ratio": args.mlp_ratio,
                    "attn_drop": args.attn_drop,
                    "drop": args.drop,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "end_iter": args.end_iter,
                    "search_iter": args.search_iter,
                    "rd_bands": args.rd_bands,
                    "rd_rows": args.rd_rows,
                    "rd_cols": args.rd_cols,
                    "seed": seed,
                    "cuda_visible_devices": args.cuda_visible_devices,
                },
                "outputs": {
                    "heatmap": str(save_subdir / "GT-HAD_map.mat"),
                    "roc": str(save_subdir / "GT-HAD_roc.mat"),
                },
            }

            if args.report_json:
                report_path = Path(args.report_json)
                report_path.parent.mkdir(parents=True, exist_ok=True)
                with report_path.open("w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)

            return summary


if __name__ == "__main__":
    args = parse_args()
    run_one(args.file, args)
