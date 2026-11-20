"""
All the loss criteria that were tried. Every criterion takes (sample dict, model output) and returns one loss value per
sample [BS] (the mean is taken in the training loop). The sinkhorn one returns a tuple (total, TP, FN, FP).
"""

import numpy as np
import torch
import torch.nn.functional as F


### phase 1: one leak --------------------------------------------------------------------------------------------------
def distance_loss(sample, pred):
    # take euclidean distance between leak and estimated leak.
    return torch.norm(sample["pos_leaks"].squeeze(1) - pred, p=2, dim=1)


### phase 2: slots + soft permutation ----------------------------------------------------------------------------------
def my_sinkhorn_norm(mtx_cost, temp=0.01, n_iter=20):
    # mtx_cost is the cost matrix (= "logits" / scores) and should ideally be square for this operation to make sense
    # a kernel in log space is used to do the calculations and is only exponentiated at the end to get probabilities!
    # minus because smaller cost should yield higher probability...
    mtx_kernel = -mtx_cost / temp

    for _ in range(n_iter):
        mtx_kernel = mtx_kernel - torch.logsumexp(mtx_kernel, dim=-1, keepdim=True)
        mtx_kernel = mtx_kernel - torch.logsumexp(mtx_kernel, dim=-2, keepdim=True)

    return torch.exp(mtx_kernel)


def matching_loss(sample, pred):
    """
    n-Leaks, intricate soft permutation algorithm with FN & FP punishing.

    Every predicted slot has a confidence and a position. Slots are softly assigned to the (padded) label slots by
    sinkhorn-normalizing a cost matrix. For a confident slot the cost of a label is its distance to it, for an
    unconfident slot real labels are expensive and padding labels are free.

    sample: the whole sample dict
    pred:   model output of shape [BS, n_leak_max, 3]
    returns (lTOT, lTP, lFN, lFP), each [BS]
    """
    BS, DIM = pred.shape[0], pred.shape[1]
    device = pred.device

    ### extract different elements
    pred_conf = torch.sigmoid(pred[:, :, 0])  # [BS, n_leak_max]
    pred_pos = pred[:, :, [1, 2]]  # [BS, n_leak_max, 2]
    label_msk = sample["msk_leaks"]  # [BS, n_leak_max]
    label_pos = sample["pos_leaks"]  # [BS, n_leak_max, 2]

    ### calculate euclidean distance matrix [BS, n_leak_max (pred), n_leak_max (label)]
    euclid_dist = torch.zeros(BS, DIM, DIM, 2, device=device)
    euclid_dist += pred_pos[:, :, None, :]
    euclid_dist -= label_pos[:, None, :, :]
    euclid_dist = torch.norm(euclid_dist, p=2, dim=3, keepdim=False)
    euclid_dist = euclid_dist.masked_fill(label_msk[:, None, :] == 1, 0)

    # calculate a "soft maximum" of all the euclidean distances
    temp = 0.05
    pseudomax = F.softmax(euclid_dist.view(-1, DIM * DIM) / temp, dim=1).view(-1, DIM, DIM)
    pseudomax = torch.sum(pseudomax * euclid_dist, dim=(1, 2))

    # put together the cost kernel
    cost_kernel = pseudomax[:, None, None] * torch.ones(BS, DIM, DIM, device=device)
    cost_kernel = (1 - pred_conf[:, :, None]) * cost_kernel * (1 - label_msk[:, None, :])
    cost_kernel += pred_conf[:, :, None] * euclid_dist

    # apply sinkhorn normalization
    permutation = my_sinkhorn_norm(cost_kernel, temp=0.1, n_iter=20)

    ### assemble loss
    # expected distance between two random points in a unit square, as the price of a missed / hallucinated leak
    expdist = (2 + np.sqrt(2) + 5 * np.log(np.sqrt(2) + 1)) / 15

    lTP = torch.sum(euclid_dist * permutation, dim=(1, 2)) / torch.sum(1 - label_msk, dim=1)

    lFN = (1 - pred_conf[:, :, None]) * torch.ones(BS, DIM, DIM, device=device) * (1 - label_msk[:, None, :])
    lFN = torch.sum(lFN * permutation, dim=(1, 2)) * expdist

    lFP = (pred_conf[:, :, None]) * torch.ones(BS, DIM, DIM, device=device) * (label_msk[:, None, :])
    lFP = torch.sum(lFP * permutation, dim=(1, 2)) * expdist

    W_TP = 2
    W_FN = 1
    W_FP = 2
    lTOT = W_TP * lTP + W_FN * lFN + W_FP * lFP

    return lTOT, lTP, lFN, lFP


### phase 2: heatmaps --------------------------------------------------------------------------------------------------
def generate_covar(s1: torch.Tensor, s2: torch.Tensor, alpha: torch.Tensor):
    """
    creates valid covariance matrices from 2 eigenvalues (positive!) and one orientation (0-2pi)

    Args:
        s1    (torch.Tensor): first singular value  [BS, nBlobs]
        s2    (torch.Tensor): second singular value [BS, nBlobs]
        alpha (torch.Tensor): angle in rad          [BS, nBlobs]

    Returns:
        covar (torch.Tensor): [BS, nBlobs, 2, 2]
    """

    # create the diagonal matrix from the singular values => [BS, nBlobs, 2, 2]
    sigma = torch.concat([s1[:, :, None], s2[:, :, None]], dim=2)
    sigma = torch.diag_embed(sigma)

    # create the rotation matrices from angle alpha => [BS, nBlobs, 2, 2]
    sin_alpha = torch.sin(alpha)
    cos_alpha = torch.cos(alpha)
    rotmat = torch.concat(
        [
            torch.concat([cos_alpha[:, :, None], -sin_alpha[:, :, None]], dim=2)[:, :, :, None],
            torch.concat([sin_alpha[:, :, None], cos_alpha[:, :, None]], dim=2)[:, :, :, None],
        ],
        dim=3,
    )

    # produce covariance mat by: covar = R.T @ Σ @ R (broadcasting over BS and nBlobs)
    covar = torch.einsum("BNji, BNjk, BNkl -> BNil", rotmat, sigma, rotmat)
    return covar


def batched_2D_gaussian(covar: torch.Tensor, mean: torch.Tensor, stren: torch.Tensor, POS: torch.Tensor, norm=False):
    """
    returns the 2D gaussian (=Z) at position (X, Y) over the whole 2D grid.
    Uses the formula 1/(2pi * |Σ|**0.5) * exp[-0.5 * (x-μ).T @ Σ^(-1) @ (x-μ)]

    Args:
        covar (torch.Tensor): [BS, nBlobs, 2, 2]
        mean  (torch.Tensor): [BS, nBlobs, 2]
        stren (torch.Tensor): [BS, nBlobs]
        POS   (torch.Tensor): [szX, szY, 2]

    Returns:
        pdfs  (torch.Tensor): [BS, nBlobs, szX, szY]
    """
    device = mean.device

    # subtract mean to get deviations => [BS, nBlobs, szX, szY, 2]
    dev = POS[None, None, :, :, :].to(device) - mean[:, :, None, None, :]

    # regularize (for stability) and invert covariances => [BS, nBlobs, 2, 2]
    epsilon = 10**-5
    covar_inv = covar + epsilon * torch.eye(2, device=device)[None, None, :, :]
    covar_inv = torch.inverse(covar_inv)

    # calculate the exponent (= mahalanobis distance) with dev.T @ Σ.inv @ dev. Watch out, when broadcasting formula,
    # doing dev.T @ Σ.inv @ dev is actually wrong, as for the second @, only the diagonal of the result would be
    # needed! (many unsensible coordinate combinations are calculated in the off-diagonals this way)
    mahalanobi = torch.einsum("BNxyi, BNij, BNxyj -> BNxy", dev, covar_inv, dev)

    if norm is True:
        # calculate the rest of the equation and output normalized gaussians (true pdfs)
        norm_factor = 1 / (2 * torch.pi * torch.sqrt(torch.det(covar)))
        pdfs = torch.einsum("BN, BNxy -> BNxy", norm_factor, torch.exp(-0.5 * mahalanobi))
    else:
        # output non-normalized functions (max of 1)
        pdfs = torch.exp(-0.5 * mahalanobi)

    # scale pdfs by strength scalar
    pdfs = torch.einsum("BN, BNxy -> BNxy", stren, pdfs)

    return pdfs


def softmaxadd_pdfs(pdfs: torch.Tensor, softmax_temp: float = 0.1):
    # combine all blobs of one scene into one map with a soft maximum over the blobs
    soft_argmax = torch.softmax(pdfs / softmax_temp, dim=1)
    pdfs_added = torch.einsum("BNxy, BNxy -> Bxy", soft_argmax, pdfs)

    return pdfs_added


def make_grid(grid_res, grid_lim=(-1, 1)):
    x, y = torch.linspace(grid_lim[0], grid_lim[1], grid_res), torch.linspace(grid_lim[0], grid_lim[1], grid_res)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    POS = torch.concat([X[:, :, None], Y[:, :, None]], dim=2)
    return POS


def label_heatmap(sample, grid_res, covarboost=1):
    """
    The ground truth map: one gaussian blob at each leak location. The blobs are shaped like the wind covariance
    matrix of the scene (covarboost makes them bigger / smoother). -> [BS, grid_res, grid_res]
    """
    POS = make_grid(grid_res)

    label_strength = 1 - sample["msk_leaks"]  # [BS, nBlobs]
    label_mean = sample["pos_leaks"]  # [BS, nBlobs, 2]
    n_max_leaks = sample["msk_leaks"].shape[1]
    label_covar = torch.tile(sample["wind_covar"][:, None, :, :], (1, n_max_leaks, 1, 1))  # [BS, nBlobs, 2, 2]

    # get LABEL pdf over grid and soft-max-add them together (substitute for max)
    label_pdfs = batched_2D_gaussian(covarboost * label_covar, label_mean, label_strength, POS)
    label_pdf = softmaxadd_pdfs(label_pdfs, softmax_temp=0.1)

    return label_pdf


def blob_heatmap(pred):
    """turns the [BS, nBlobs, 6] output of the blob head into the predicted map [BS, 100, 100]"""
    POS = make_grid(100)

    pred_strength = pred[:, :, 0]  # [BS, nBlobs]
    pred_mean = pred[:, :, [1, 2]]  # [BS, nBlobs, 2]
    pred_covar_param = pred[:, :, [3, 4, 5]]  # [BS, nBlobs, 3]

    # add nonlinearity to raw outputs to ensure positive singular values and positive pred_strength (what about mean??)
    pred_strength = torch.sigmoid(pred_strength)
    pred_covar_param[:, :, [0, 1]] = torch.sigmoid(pred_covar_param[:, :, [0, 1]])
    pred_mean = torch.clamp(pred_mean, -1, 1)

    # generate all the covariances => [BS, nBlobs, 2, 2]
    pred_covar = generate_covar(
        pred_covar_param[:, :, 0], pred_covar_param[:, :, 1], 2 * np.pi * pred_covar_param[:, :, 2]
    )

    # get PREDICTED pdf over grid and soft-max-add them together (substitute for max)
    pred_pdfs = batched_2D_gaussian(pred_covar, pred_mean, pred_strength, POS)
    pred_pdf = softmaxadd_pdfs(pred_pdfs, softmax_temp=0.1)

    return pred_pdf


def blob_heatmap_loss(sample, pred):
    """
    n-Leaks, predicting confidence heatmap, by parametric gaussian blobs. RMSE for the pixel-wise difference between
    the predicted field and the true field on a 100x100 grid.

    pred: [BS, nBlobs, 6]
    """
    grid_res = 100
    pred_pdf = blob_heatmap(pred)
    label_pdf = label_heatmap(sample, grid_res)

    diff = label_pdf - pred_pdf
    rmse = torch.sqrt(torch.sum(diff**2, dim=(1, 2)) / grid_res**2)

    return rmse


def weighted_heatmap_loss(sample, pred):
    """
    n-Leaks, predicting confidence heatmap, convolution + upsampling. Foreground / background masking and weighting is
    used to balance the two classes kind of.

    pred: prediction field [BS, 64, 64] (logits)
    """
    grid_res = 64
    pred_pdf = torch.sigmoid(pred)

    # watch out, if the covariances are boosted too much, the foreground / background masking can become nonsensical
    # and lead to NaNs in the losses!
    covarboost = 1
    label_pdf = label_heatmap(sample, grid_res, covarboost)

    # weight for balancing
    foreground_thresh = 0.05
    thresh = label_pdf > foreground_thresh
    total = torch.sum(thresh, dim=(1, 2))
    wgh = total / (grid_res**2)
    mask = thresh / (wgh[:, None, None]) + ~thresh / (1 - wgh[:, None, None])

    # construct weighted rmse between fields
    diff_wgh = (label_pdf - pred_pdf) ** 2 * mask
    rmse = torch.sqrt(torch.sum(diff_wgh, dim=(1, 2)) / (grid_res**2))

    # add output regularization (0.001 is just to match magnitude!)
    # (switched off: I guess this is a bit desperate haha and doesn't really help)
    λ = 0.001 * 0
    output_sum = torch.sum(pred_pdf, dim=(1, 2))

    return rmse + λ * output_sum


def plain_heatmap_loss(sample, pred):
    """
    Just for checking if the model is even able to overfit. No fancy class-equalization weighting etc...

    pred: prediction field [BS, 32, 32] (logits)
    """
    grid_res = 32
    pred_pdf = torch.sigmoid(pred)

    covarboost = 2
    label_pdf = label_heatmap(sample, grid_res, covarboost)

    diff = label_pdf - pred_pdf
    rmse = torch.sqrt(torch.sum(diff**2, dim=(1, 2)) / (grid_res**2))

    return rmse


### phase 2: sanity check ----------------------------------------------------------------------------------------------
def count_loss(sample, pred):
    # loss is just the absolute difference to the true nr of leaks
    return torch.abs(sample["n_leaks"] - pred)
