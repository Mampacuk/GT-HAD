# lsad_cr_idw.py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat, savemat
import time
import math

def fun_LSAD_CR_IDW(hsi_np, win_out, win_in, lambd, device=None, verbose=True):
    """
    PyTorch implementation of your MATLAB fun_LSAD_CR_IDW.
    hsi_np: numpy array (rows, cols, bands)
    win_out: outer window size (odd)
    win_in: inner window size (odd)
    lambd: lambda regularization scalar
    device: 'cuda' or 'cpu' or None -> auto
    Returns RDet as numpy array (rows, cols)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # sizes
    rows, cols, bands = hsi_np.shape
    t = win_out // 2
    t1 = win_in // 2
    M = win_out * win_out

    # Normalize (same as MATLAB)
    flat = hsi_np.reshape(-1, bands).astype(np.float32)
    maxVal = flat.max()
    minVal = flat.min()
    if maxVal == minVal:
        hsi_norm = np.zeros_like(hsi_np, dtype=np.float32)
    else:
        hsi_norm = ((hsi_np.astype(np.float32) - minVal) / (maxVal - minVal)).astype(np.float32)

    # Move to torch: shape N=1, C=bands, H, W
    x = torch.from_numpy(hsi_norm).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, bands, rows, cols)

    # Pad with reflection so patches near border behave like your mirrored DataTest
    pad_total = t + t1  # pad so we can take shifted center patches easily
    x_padded = F.pad(x, (pad_total, pad_total, pad_total, pad_total), mode='reflect')  # (1, bands, H+2pad, W+2pad)
    _, C, Hpad, Wpad = x_padded.shape

    # Use unfold to get all win_out patches from padded image (no extra padding)
    unfold = torch.nn.Unfold(kernel_size=win_out, stride=1, padding=0)
    patches_all = unfold(x_padded)  # shape (1, bands * M, L), L = (Hpad - win_out +1)*(Wpad - win_out +1)
    # L equals (rows + 2*pad_total - 2*t) * (cols + 2*pad_total - 2*t) = rows + 2*t1 ... etc
    # We'll index into columns corresponding to the centers we need.

    # Precompute indices for centers inside padded grid
    # The column index for a patch whose top-left corner is at (y0,x0) is: idx = y0 * (Wpad - win_out + 1) + x0
    stride_cols = Wpad - win_out + 1

    # Precompute linear indices in the patches_all columns for all possible center positions
    # A patch centered at padded coords (cy, cx) corresponds to top-left at (cy - t, cx - t)
    def center_to_patch_col_index(cy, cx):
        y0 = cy - t
        x0 = cx - t
        return y0 * stride_cols + x0

    # Precompute the flattened positions within an M-size patch for which we KEEP (exclude inner window).
    # ordering inside unfold is [band0 positions(0..M-1), band1 positions(0..M-1), ...]
    keep_positions = []
    for r in range(win_out):
        for cpos in range(win_out):
            if not (t - t1 <= r <= t + t1 and t - t1 <= cpos <= t + t1):
                keep_positions.append(r * win_out + cpos)
    keep_positions = np.array(keep_positions, dtype=np.int64)  # length num_S

    # Precompute IDW weights (distance-based) for the relative positions; we'll pick only those corresponding to keep_positions
    # Build the distance map for win_out x win_out and mask out inner window
    coords_x = np.arange(-t, t + 1)
    coords_y = np.arange(t, -t - 1, -1)  # note MATLAB used descending Y
    XX, YY = np.meshgrid(coords_x, coords_y)
    Dd = np.sqrt(XX.astype(np.float32) ** 2 + YY.astype(np.float32) ** 2)  # shape (win_out, win_out)
    # mask center inner window
    inner_r0 = t - t1
    inner_r1 = t + t1
    Dd[inner_r0:inner_r1 + 1, inner_r0:inner_r1 + 1] = np.nan
    IDW_vec = Dd.reshape(M, 1)
    IDW_vec = IDW_vec[~np.isnan(IDW_vec)].reshape(-1)  # length num_S
    # convert to IDW weighting as in MATLAB: w = 1/d^2 normalized
    SumW = np.sum((IDW_vec ** -2))
    IDW_vec = (IDW_vec ** -2) / SumW
    IDW = torch.from_numpy(IDW_vec.astype(np.float32)).to(device)  # (num_S,)

    # Precompute keep_positions for selecting bands
    keep_pos_torch = torch.from_numpy(keep_positions).to(device)  # indices in 0..M-1

    # Precompute patch columns indices offsets for inner shifts relative to a center column
    # The inner shift centers relative offsets (ki, kj)
    inner_shifts = [(ki, kj) for ki in range(-t1, t1 + 1) for kj in range(-t1, t1 + 1)]
    S = len(inner_shifts)  # should equal win_in^2

    # For progress
    total_to_process = rows * cols
    processed = 0
    percent_step = 1
    current_percent_goal = percent_step
    tstart = time.time()

    RDet = np.zeros((rows, cols), dtype=np.float32)

    # Prepare identity template for adding diagonal reg later; we'll use torch.diag_embed
    eye_template = None  # created per-batch as needed since size num_S is small

    # Pre-calc the patches_all as a view for faster indexing
    # patches_all: (1, bands*M, L)
    # reshape to (bands, M, L)
    _, BM, L = patches_all.shape
    assert BM == C * M
    patches_all = patches_all.view(1, C, M, L)[0]  # (bands, M, L)

    # Convenience: a mapping to quickly compute patch column index for any padded (cy,cx):
    # valid center y range in padded: from pad_total to pad_total + rows -1
    # We'll iterate center positions in original image order r=0..rows-1, c=0..cols-1
    for r in range(rows):
        # compute padded center row coordinate
        cy0 = pad_total + r
        for ccol in range(cols):
            cx0 = pad_total + ccol

            # compute col indices in patches_all corresponding to inner shifts of centers
            # Each shift center has padded coords (cy0 + ki, cx0 + kj)
            col_indices = []
            for (ki, kj) in inner_shifts:
                cy = cy0 + ki
                cx = cx0 + kj
                idx = center_to_patch_col_index(cy, cx)
                col_indices.append(idx)
            col_indices = torch.tensor(col_indices, dtype=torch.long, device=device)  # (S,)

            # Extract patches for all shifts: shape (bands, M, S)
            sel = patches_all[:, :, col_indices]  # (bands, M, S)
            # Select keep positions and rearrange to H: (S, num_S, bands)
            # sel[:, keep_positions, :] -> (bands, num_S, S)
            sel_keep = sel[:, keep_pos_torch, :]  # (bands, num_S, S)
            H_batch = sel_keep.permute(2, 1, 0).contiguous()  # (S, num_S, bands)

            # Center pixel spectrum (bands,)
            CenPix = x[0, :, r, ccol].to(device)  # (bands,)

            # Compute temp = CenPix - H' for each shift:
            # H_batch: (S, num_S, bands)
            # Compute norms per sample: norms (S, num_S)
            temp_diff = CenPix.view(1, 1, bands) - H_batch  # broadcasting => (S, num_S, bands)
            norms = torch.linalg.norm(temp_diff, dim=2)  # (S, num_S)

            # Build G = H * H' + lambda * diag( (IDW * norms)^2 )
            # First compute HHT = (S, num_S, num_S)
            # we can compute via batch matmul: H_batch @ H_batch.transpose(-1,-2)
            HHT = torch.matmul(H_batch, H_batch.transpose(-1, -2))  # (S, num_S, num_S)

            # diag entries:
            # IDW (num_S,) -> broadcast to (S, num_S)
            diag_entries = (IDW.unsqueeze(0) * norms) ** 2  # (S, num_S)
            # add lambda * diag_entries on diagonal
            # create diagonal matrices
            reg_diag = lambd * diag_entries  # (S, num_S)
            # add diagonal: use diag_embed then add
            G = HHT + torch.diag_embed(reg_diag)  # (S, num_S, num_S)

            # Right-hand side: H * CenPix -> shape (S, num_S)
            H_Cen = torch.einsum('snb,b->sn', H_batch, CenPix)  # (S, num_S)
            H_Cen = H_Cen.unsqueeze(-1)  # (S, num_S, 1)

            # Batched pseudo-inverse
            # torch.linalg.pinv supports batched input
            # pinv(G): (S, num_S, num_S)
            try:
                G_pinv = torch.linalg.pinv(G)  # batched
            except Exception:
                # fallback to torch.pinverse if older version
                G_pinv = torch.pinverse(G)

            # W = pinv(G) @ H_Cen -> (S, num_S, 1)
            W = torch.matmul(G_pinv, H_Cen)  # (S, num_S, 1)

            # Reconstruction: H' * W -> (S, bands, 1)
            recon = torch.matmul(H_batch.transpose(-1, -2), W).squeeze(-1)  # (S, bands)

            # DetY per shift: norm(CenPix - recon)
            diff_rec = CenPix.view(1, bands) - recon  # (S, bands)
            DetY = torch.linalg.norm(diff_rec, dim=1)  # (S,)

            # Sum over all inner shifts to produce RDet at this center
            RDet[r, ccol] = DetY.sum().detach().cpu().item()

            # progress
            processed += 1
            if verbose:
                percent_processed = processed / total_to_process * 100.0
                if percent_processed >= current_percent_goal:
                    elapsed = time.time() - tstart
                    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: Processed {percent_processed:.2f}% (LSAD-CR-IDW) elapsed {elapsed:.1f}s")
                    current_percent_goal += percent_step

    return RDet


# ---------------- small main ----------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run LSAD-CR-IDW (GPU-accelerated PyTorch)")
    parser.add_argument("--matfile", type=str, default="input.mat", help="MAT file that contains variable 'hsi'")
    parser.add_argument("--varname", type=str, default="hsi", help="Variable name in MAT file (default 'hsi')")
    parser.add_argument("--out", type=str, default="RDet_out.mat", help="Output MAT filename")
    parser.add_argument("--inner", type=int, default=25, help="INNER_WINDOW_SIZE")
    parser.add_argument("--outer", type=int, default=27, help="OUTER_WINDOW_SIZE")
    parser.add_argument("--lambda_", type=float, default=100.0, help="LAMBDA")
    parser.add_argument("--device", type=str, default=None, help="device: 'cuda' or 'cpu' (auto if None)")
    args = parser.parse_args()

    print("Loading MAT file:", args.matfile)
    mat = loadmat(args.matfile)
    if args.varname not in mat:
        raise KeyError(f"Variable '{args.varname}' not found in {args.matfile}. Available keys: {list(mat.keys())}")
    hsi = mat[args.varname].astype(np.float64)
    print("HSI shape:", hsi.shape)

    RDet = fun_LSAD_CR_IDW(hsi, win_out=args.outer, win_in=args.inner, lambd=args.lambda_, device=args.device)
    print("Saving result to", args.out)
    savemat(args.out, {"RDet": RDet})
    print("Done.")
