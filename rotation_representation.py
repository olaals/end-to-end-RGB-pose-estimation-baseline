import torch
import pytorch3d as pt3d
from pytorch3d import transforms as pt3dtf
import math

def compute_rotation_matrix_from_ortho6d(poses):
    """
    Code from https://github.com/papagina/RotationContinuity
    On the Continuity of Rotation Representations in Neural Networks
    Zhou et al. CVPR19
    https://zhouyisjtu.github.io/project_rotation/rotation.html
    """
    assert poses.shape[-1] == 6
    x_raw = poses[..., 0:3]
    y_raw = poses[..., 3:6]
    x = x_raw / torch.norm(x_raw, p=2, dim=-1, keepdim=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.norm(z, p=2, dim=-1, keepdim=True)
    y = torch.cross(z, x, dim=-1)
    matrix = torch.stack((x, y, z), -1)
    return matrix


def symmetric_orthogonalization(x):
  """
  Code from https://github.com/amakadia/svd_for_pose

  Maps 9D input vectors onto SO(3) via symmetric orthogonalization.

  x: should have size [batch_size, 9]

  Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).
  """
  m = x.view(-1, 3, 3)
  u, s, v = torch.svd(m)
  vt = torch.transpose(v, 1, 2)
  det = torch.det(torch.matmul(u, vt))
  det = det.view(-1, 1, 1)
  vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
  r = torch.matmul(u, vt)
  return r

def so3_exp_map(
    log_rot: torch.Tensor, eps: float = 0.0001
    ):
    """
    Copied from Pytorch3D : https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/so3.html#so3_exp_map
    A helper function that computes the so3 exponential map and,
    apart from the rotation matrix, also returns intermediate variables
    that can be re-used in other functions.
    """
    _, dim = log_rot.shape
    if dim != 3:
        raise ValueError("Input tensor shape has to be Nx3.")

    nrms = (log_rot * log_rot).sum(1)
    # phis ... rotation angles
    rot_angles = torch.clamp(nrms, eps).sqrt()
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = hat(log_rot)
    skews_square = torch.bmm(skews, skews)

    R = (
        # pyre-fixme[16]: `float` has no attribute `__getitem__`.
        fac1[:, None, None] * skews
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    return R

def hat(v: torch.Tensor) -> torch.Tensor:
    """
    Copied from Pytorch3D:  https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/so3.html#so3_exp_map
    Compute the Hat operator [1] of a batch of 3D vectors.

    Args:
        v: Batch of vectors of shape `(minibatch , 3)`.

    Returns:
        Batch of skew-symmetric matrices of shape
        `(minibatch, 3 , 3)` where each matrix is of the form:
            `[    0  -v_z   v_y ]
             [  v_z     0  -v_x ]
             [ -v_y   v_x     0 ]`

    Raises:
        ValueError if `v` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    h = torch.zeros((N, 3, 3), dtype=v.dtype, device=v.device)

    x, y, z = v.unbind(1)

    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x
    return h

def vec_3d_to_SO3(x):
    """
    Based on the Lie Group mapping from the minimal vector space to the SO3 manifold (rot mat)
    """
    device = x.device
    bsz = x.shape[0]
    assert x.shape == (bsz,3)

    rot_mats = so3_exp_map(x)
    return rot_mats












def calculate_T_CO_pred(model_output, T_CO_init, rot_repr, Ks):
    bsz = model_output.shape[0]
    if rot_repr == 'SVD':
        assert model_output.shape == (bsz, 12)
    elif rot_repr == '6D':
        assert model_output.shape == (bsz, 9)
    elif rot_repr == '3D':
        assert model_output.shape == (bsz, 6)

    assert T_CO_init.shape == (bsz, 4,4)
    assert Ks.shape == (bsz, 3,3)
    

    device = model_output.device
    bsz = T_CO_init.shape[0]
    T_CO_pred = torch.ones((bsz, 4, 4)).to(device)
    if rot_repr == 'SVD':
        dR = symmetric_orthogonalization(model_output[:, 0:9])
        vxvyvz = model_output[:, 9:12]
    elif rot_repr == '6D':
        dR = compute_rotation_matrix_from_ortho6d(model_output[:, 0:6])
        vxvyvz = model_output[:, 6:9]
    elif rot_repr == '3D':
        dR = vec_3d_to_SO3(model_output[:, 0:3])
        vxvyvz = model_output[:, 3:6]
    else:
        assert False
    vx = vxvyvz[:, 0]
    vy = vxvyvz[:, 1]
    vz = vxvyvz[:, 2]
    #vz = 1+nn.Tanh()(vz)*0.2

    R_k = T_CO_init[:, :3, :3]

    K = Ks[0]

    fx = K[0,0]
    fy = K[1,1]
    z_k = T_CO_init[:, 2, 3]
    z_kp1 = vz*z_k
    x_k = T_CO_init[:, 0, 3]
    y_k = T_CO_init[:, 0, 3]
    x_kp1 = (vx/fx + x_k/z_k)*z_kp1
    y_kp1 = (vy/fy + y_k/z_k)*z_kp1
    R_k = R_k.float()
    R_kp1 = torch.einsum('bij,bjk->bik', dR, R_k)

    ## assemble
    T_CO_pred[:, :3, :3] = R_kp1
    T_CO_pred[:, 0, 3] = x_kp1
    T_CO_pred[:, 1, 3] = y_kp1
    T_CO_pred[:, 2, 3] = z_kp1
    T_CO_pred[:, 3, :3] = torch.tensor(0)
    return T_CO_pred


if __name__ == '__main__':
    tens = torch.zeros((2,3))
    tens[0][0] = math.pi/2
    tens[0][1] = 0.0
    tens[0][2] = 0.0

    rots = vec_3d_to_SO3(tens)
    print(rots)
    

