import spatialmath as sm
import numpy as np


def look_at_SE3(origin, target, up):
    """
    Useful for positioning the camera in a scene. 
    Origin: The position of the camera, for example [3,3,3] means that the x,y,z location of the camera is at x=3, y=3, z=3
    Target: The point in space the camera 'looks' at
    Up: The direction in world coordinates that defines which "roll" the camera has. Usually [0,0,1]
    """
    assert len(origin) == 3
    assert len(target) == 3
    assert len(up) == 3
    origin= np.array(origin)
    target = np.array(target)
    up = np.array(up)
    z = target - origin
    z = z/np.linalg.norm(z)
    x = np.cross(up, z)
    x = -x/np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.array([x,y,z]).transpose()
    so3_sm = sm.SO3(R, check=True)
    se3_sm = sm.SE3.Rt(so3_sm, origin)
    return se3_sm

def get_random_z_rot():
    z_rot = np.random.uniform(0.0, 180.0)
    T_zrot = sm.SE3.Rz(z_rot, unit='deg')
    return T_zrot

def get_random_unit_axis():
    # not perfectly random, as points are sampled uniformly in a cube then normalized to unit length
    random_axis = np.random.random(3)-0.5
    random_unit_axis = random_axis/np.linalg.norm(random_axis)
    return random_unit_axis

def apply_small_random_rotation_translation(T, theta_range_deg, xyz_transl_range, sampling="uniform"):
    if sampling == "uniform":
        theta = np.random.uniform(-theta_range_deg, theta_range_deg)
        x_transl = np.random.uniform(-xyz_transl_range, xyz_transl_range)
        y_transl = np.random.uniform(-xyz_transl_range, xyz_transl_range)
        z_transl = np.random.uniform(-xyz_transl_range, xyz_transl_range)
    if sampling == "normal":
        theta = np.random.normal(0, theta_range_deg)
        x_transl = np.random.normal(0, xyz_transl_range)
        y_transl = np.random.normal(0, xyz_transl_range)
        z_transl = np.random.normal(0, xyz_transl_range)


    rotation_axis = get_random_unit_axis()
    rotation_SO3 = sm.SO3.AngleAxis(theta,rotation_axis, unit='deg')
    transl = np.array([x_transl, y_transl, z_transl])
    delta_SE3 = sm.SE3.Rt(rotation_SO3, transl)
    new_SE3 = delta_SE3*T
    return new_SE3

def get_random_rotation_translation(xyz_transl_range):
    theta = np.random.uniform(0, 360)
    rotation_axis = get_random_unit_axis()
    rotation_SO3 = sm.SO3.AngleAxis(theta,rotation_axis, unit='deg')
    x_transl = np.random.normal(0, xyz_transl_range)
    y_transl = np.random.normal(0, xyz_transl_range)
    z_transl = np.random.normal(0, xyz_transl_range)
    transl = np.array([x_transl, y_transl, z_transl])
    SE3_mat = sm.SE3.Rt(rotation_SO3, transl)
    return SE3_mat

def get_T_CW(base_distance, random_deviation=0.0):
    distance = base_distance + np.random.uniform(-random_deviation, random_deviation)
    T_WC = look_at_SE3([distance, 0, 0], [0,0,0], [0,0,1])
    T_CW = T_WC.inv()
    return T_CW

def get_T_CO_init_and_gt(scene_config):
    dist_CW = scene_config["distance_cam_to_world"]
    dist_CW_dev = scene_config["distance_cam_to_world_deviation"]
    WO_gt_transl_dev = scene_config["world_to_object_gt_transl_deviation"]
    WO_transl_dev = scene_config["world_to_object_transl_deviation"]
    WO_angle_dev = scene_config["world_to_object_angle_deviation"]
     
    T_WO_gt = get_random_rotation_translation(WO_gt_transl_dev)
    T_WO_init_guess = apply_small_random_rotation_translation(T_WO_gt, WO_angle_dev, WO_transl_dev)
    T_CW = get_T_CW(dist_CW, dist_CW_dev)
    T_CO_gt = T_CW*T_WO_gt
    T_CO_init_guess = T_CW*T_WO_init_guess
    return T_CO_init_guess, T_CO_gt




if __name__ == '__main__':
    config = get_config()
    T_CO_init, T_CO_gt = get_T_CO_init_and_gt(config)
    print(T_CO_init)
    print(T_CO_gt)




