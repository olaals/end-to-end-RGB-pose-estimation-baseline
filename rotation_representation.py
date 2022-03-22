

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

def calculate_T_CO_pred(model_output, T_CO_init):
    device = model_output.device
    bsz = T_CO_init.shape[0]
    T_CO_pred = torch.ones((bsz, 4, 4)).to(device)
    dR = symmetric_orthogonalization(model_output[:, 0:9])
    vxvyvz = model_output[:, 9:12]
    vx = vxvyvz[:, 0]
    vy = vxvyvz[:, 1]
    vz = vxvyvz[:, 2]
    vz = 1+nn.Tanh()(vz)*0.2
    K = get_K(device)

    R_k = T_CO_init[:, :3, :3]

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

    # assemble
    T_CO_pred[:, :3, :3] = R_kp1
    T_CO_pred[:, 0, 3] = x_kp1
    T_CO_pred[:, 1, 3] = y_kp1
    T_CO_pred[:, 2, 3] = z_kp1
    T_CO_pred[:, 3, :3] = torch.tensor(0)
    return T_CO_pred

