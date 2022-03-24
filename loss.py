from rotation_representation import calculate_T_CO_pred

def compute_ADD_L1_loss(TCO_gt, TCO_pred, points):
    """
    copied from
    https://github.com/ylabbe/cosypose/blob/master/cosypose/lib3d/mesh_losses.py
    """

    bsz = len(TCO_gt)
    assert TCO_pred.shape == (bsz, 4, 4) and TCO_gt.shape == (bsz, 4, 4)
    assert points.dim() == 3 and points.shape[-1] == 3
    dists = (transform_pts(TCO_gt, points) - transform_pts(TCO_pred, points)).abs().mean(dim=(-1, -2))
    return dists.mean()



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






