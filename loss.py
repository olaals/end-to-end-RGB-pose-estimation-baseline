
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


