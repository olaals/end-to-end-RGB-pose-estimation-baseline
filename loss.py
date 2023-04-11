import torch
from rotation_representation import calculate_T_CO_pred
import numpy as np

def compute_ADD_L1_loss(TCO_gt, TCO_pred, points, use_batch_mean=True):
    """
    copied from
    https://github.com/ylabbe/cosypose/blob/master/cosypose/lib3d/mesh_losses.py
    """

    bsz = len(TCO_gt)
    assert TCO_pred.shape == (bsz, 4, 4) and TCO_gt.shape == (bsz, 4, 4)
    assert points.dim() == 3 and points.shape[-1] == 3
    dists = (transform_pts(TCO_gt, points) - transform_pts(TCO_pred, points)).abs().mean(dim=(-1, -2))
    if use_batch_mean:
        return dists.mean()
    else:
        return dists

def compute_ADD_L2_loss(TCO_gt, TCO_pred, points, use_batch_mean=True):
    bsz = len(TCO_gt)
    assert TCO_pred.shape == (bsz, 4, 4) and TCO_gt.shape == (bsz, 4, 4)
    assert points.dim() == 3 and points.shape[-1] == 3
    square_dist = torch.square(transform_pts(TCO_gt, points) - transform_pts(TCO_pred, points))
    print(square_dist.shape)
    sum_dists = torch.sum(square_dist, axis=2)
    dists = torch.sqrt(sum_dists)
    dists = torch.mean(dists, axis=1).flatten()
    print(dists.shape)
    if use_batch_mean:
        return dists.mean()
    else:
        return dists

def compute_angular_dist(T_CO_gt, T_CO_pred):
    bz = T_CO_gt[0]
    R_CO_gt = T_CO_gt[:,:3,:3]
    R_CO = T_CO_pred[:,:3,:3]
    R_gt = R_CO_gt[0].detach().cpu().numpy()
    R_pred = R_CO[0].detach().cpu().numpy()
    diff = R_pred@np.transpose(R_gt) 
    angle = np.arccos((np.trace(diff)-1.0)/2.0)
    return angle

def compute_transl_dist(T_CO_gt, T_CO_pred):
    t_CO_gt = T_CO_gt[0,:3,3].detach().cpu().numpy()
    t_CO = T_CO_pred[0,:3,3].detach().cpu().numpy()
    diff = t_CO_gt-t_CO
    diff_sq = np.square(diff)
    summed = np.sum(diff_sq)
    sq_rt = np.sqrt(summed)
    return sq_rt







def transform_pts(T, pts):
    """
    copied from
    https://github.com/ylabbe/cosypose/blob/master/cosypose/lib3d/transform_ops.py
    """
    bsz = T.shape[0]
    n_pts = pts.shape[1]
    assert pts.shape == (bsz, n_pts, 3)
    if T.dim() == 4:
        pts = pts.unsqueeze(1)
        assert T.shape[-2:] == (4, 4)
    elif T.dim() == 3:
        assert T.shape == (bsz, 4, 4)
    else:
        raise ValueError('Unsupported shape for T', T.shape)
    pts = pts.unsqueeze(-1)
    T = T.unsqueeze(-3)
    pts_transformed = T[..., :3, :3] @ pts + T[..., :3, [-1]]
    return pts_transformed.squeeze(-1)


def compute_disentangled_ADD_L1_loss(T_CO_pred, T_CO_gt, points):
    # idea from https://github.com/ylabbe/cosypose, but more compact implementation

    disent_T_CO_rot = T_CO_gt.clone()
    disent_T_CO_depth = T_CO_gt.clone()
    disent_T_CO_transl = T_CO_gt.clone()

    disent_T_CO_rot[:, :3, :3] = T_CO_pred[:,:3,:3]
    disent_T_CO_transl[:, :2, 3] = T_CO_pred[:, :2, 3]
    disent_T_CO_depth[:, 2, 3] = T_CO_pred[:, 2, 3]

    rot_loss = compute_ADD_L1_loss(T_CO_gt, disent_T_CO_rot, points) 
    transl_loss = compute_ADD_L1_loss(T_CO_gt, disent_T_CO_transl, points) 
    depth_loss = compute_ADD_L1_loss(T_CO_gt, disent_T_CO_depth, points)
    
    disentangled_loss = rot_loss + transl_loss + depth_loss
    return disentangled_loss


# Experimental
def compute_scaled_disentl_ADD_L1_loss(T_CO_pred_prev, T_CO_pred, T_CO_gt, points):
    device = T_CO_pred.device
    disent_T_CO_rot = T_CO_gt.clone()
    disent_T_CO_depth = T_CO_gt.clone()
    disent_T_CO_transl = T_CO_gt.clone()
    T_CO_pred_prev = T_CO_pred_prev.clone().detach()

    disent_T_CO_rot[:, :3, :3] = T_CO_pred[:,:3,:3]
    disent_T_CO_transl[:, :2, 3] = T_CO_pred[:, :2, 3]
    disent_T_CO_depth[:, 2, 3] = T_CO_pred[:, 2, 3]

    rot_loss = compute_ADD_L1_loss(T_CO_gt, disent_T_CO_rot, points, False) 
    transl_loss = compute_ADD_L1_loss(T_CO_gt, disent_T_CO_transl, points, False) 
    depth_loss = compute_ADD_L1_loss(T_CO_gt, disent_T_CO_depth, points, False)

    with torch.no_grad():
        scaling = compute_ADD_L1_loss(T_CO_gt, T_CO_pred_prev, points, False)

    scaling = torch.max(scaling, torch.ones(scaling.shape).to(device)*0.01)

    disentangled_loss = ((rot_loss + transl_loss + depth_loss)/scaling).mean()
    return disentangled_loss






