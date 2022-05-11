import os
#os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import numpy as np
import spatialmath as sm
import trimesh as tm
from se3_helpers import *
import matplotlib.pyplot as plt
import pyrender
from PIL import Image
from se3_helpers import get_T_CO_init_and_gt, look_at_SE3


def get_camera_matrix(intrinsics):
    focal_len = intrinsics["focal_length"]
    img_res = intrinsics["image_resolution"]
    sensor_width = intrinsics["sensor_width"]
    pix_per_mm = sensor_width/img_res
    fx = fy = focal_len/pix_per_mm
    vx = vy = img_res/2
    K = np.array([[fx, 0, vx],[0, fy, vy],[0,0,1]])
    return K


def add_object(scene, path):
    trimesh_mesh = tm.load(path)
    mesh = pyrender.Mesh.from_trimesh(trimesh_mesh, smooth=False)
    scene.add(mesh)

def add_light(scene, T_CO):
    assert T_CO.shape == (4,4)
    T_OC = np.linalg.inv(T_CO)
    light = pyrender.SpotLight(color=np.ones(3), intensity=15.0,
                            innerConeAngle=np.pi/8.0,
                            outerConeAngle=np.pi/3.0)
    scene.add(light, pose=T_OC)

def add_camera(scene, T_CO, K):
    assert T_CO.shape == (4,4)
    T_OC = np.linalg.inv(T_CO)
    fx,fy, ux,uy = K[0,0], K[1,1], K[0,2], K[1,2]
    camera = pyrender.IntrinsicsCamera(fx, fy, ux,uy)
    scene.add(camera, pose=T_OC)

def render(scene, img_size):
    r = pyrender.OffscreenRenderer(img_size, img_size)
    color, depth = r.render(scene)
    r.delete()
    return color/255.0, depth


def render_scene(object_path, T_CO, cam_config):
    assert T_CO.shape == (4,4)
    img_size = cam_config["image_resolution"]
    if("K" not in cam_config):
        K = get_camera_matrix(cam_config)
    else:
        K = cam_config["K"]

    T_CO = sm.SE3.Rx(180, unit='deg').data[0]@T_CO # convert from OpenCV camera frame to OpenGL camera frame
    scene = pyrender.Scene()
    scene.bg_color = (0,0,0)
    add_object(scene, object_path)
    add_light(scene, T_CO)
    add_camera(scene, T_CO, K)
    img, depth = render(scene, img_size)
    return img, depth

def normalize_depth(depth_img):
    mean_val = np.mean(depth_img[depth_img>0.01])
    std = np.std(depth_img[depth_img>0.01])
    normalized = np.where(depth_img>0.01, (depth_img-mean_val)/std, 0.0)
    return normalized.astype(np.float32)


if __name__ == '__main__':
    model = "node_adapter.ply"
    T_CW = look_at_SE3([3,3,3], [0,0,0], [0,0,1])
    K = np.array([[336.43 ,0.0, 160.0 ],
            [.0, 335.045, 160.0],
            [.0,.0,1.0]])
    print(K)

    cam_config = {
        "K":K,
        "image_resolution":320
    }
    #img,d = render_scene()








