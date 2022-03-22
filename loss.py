

def get_ADD_loss(model_output, T_CO_init, T_CO_gt, verts):
    bsz = T_CO_gt.shape[0]
    assert model_output.shape == (bsz, 12)
    assert T_CO_init.shape == (bsz, 4,4)
    assert T_CO_gt.shape == (bsz, 4,4)

    T_CO_pred = calculate_T_CO_pred(model_output, T_CO_init)

    loss = compute_ADD_L1_loss(T_CO_gt.float(), T_CO_pred, verts.float())
    loss = loss.mean()

    return loss


def compute_ADD_L1_loss(TCO_gt, TCO_pred, points):
    """
    copied from
    https://github.com/ylabbe/cosypose/blob/master/cosypose/lib3d/mesh_losses.py
    """

    bsz = len(TCO_gt)
    assert TCO_pred.shape == (bsz, 4, 4) and TCO_gt.shape == (bsz, 4, 4)
    assert points.dim() == 3 and points.shape[-1] == 3
    dists = (transform_pts(TCO_gt, points) - transform_pts(TCO_pred, points)).abs().mean(dim=(-1, -2))
    return dists



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


