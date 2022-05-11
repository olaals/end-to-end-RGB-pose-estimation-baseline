import os
#os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np
import spatialmath as sm
import trimesh as tm
from se3_helpers import *
import matplotlib.pyplot as plt
import pyrender
from PIL import Image
from se3_helpers import get_T_CO_init_and_gt


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
    mesh = pyrender.Mesh.from_trimesh(trimesh_mesh)
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
    K = get_camera_matrix(cam_config)

    T_CO = sm.SE3.Rx(180, unit='deg').data[0]@T_CO # convert from OpenCV camera frame to OpenGL camera frame
    scene = pyrender.Scene()
    scene.bg_color = (0,0,0)
    add_object(scene, object_path)
    add_light(scene, T_CO)
    add_camera(scene, T_CO, K)
    img, depth = render(scene, img_size)
    return img, normalize_depth(depth)

def normalize_depth(depth_img):
    mean_val = np.mean(depth_img[depth_img>0.01])
    std = np.std(depth_img[depth_img>0.01])
    normalized = np.where(depth_img>0.01, (depth_img-mean_val)/std, 0.0)
    return normalized.astype(np.float32)


if __name__ == '__main__':
    """
    config = get_config()
    test_mesh_path = os.path.join("irrelevant-data", "airplane_0180.ply")
    #K = get_camera_matrix()
    T_OC = look_at_SE3([0.8,0.8,0.8], [0,0,0], [0,0,1])
    T_CO = T_OC.inv()
    img = render_scene(test_mesh_path, T_CO.data[0], config)
    pil_img = Image.fromarray((img*255).astype(np.uint8))
    pil_img.save(os.path.join("irrelevant-data", "airplane_0180.png"))
    plt.imshow(img)
    plt.show()
    """
    def combine_imgs(img1, img2):
        gs1 = np.mean(img1, axis=2)
        gs2 = np.mean(img2, axis=2)
        img = np.zeros((gs1.shape[0], gs1.shape[1], 3))
        img[:,:,0] = gs1
        img[:,:,1] = gs2
        return img

    import sys
    sys.path.append("configs")
    from baseline_cfg import get_config
    config = get_config()
    T_CO_init, T_CO_gt = get_T_CO_init_and_gt(config["scene_config"])
    test_mesh_path = os.path.join("irrelevant-data", "airplane_0180.ply")
    img_gt,_ = render_scene(test_mesh_path, T_CO_gt.data[0], config["camera_intrinsics"])
    img_init, _= render_scene(test_mesh_path, T_CO_init.data[0], config["camera_intrinsics"])
    fig,ax = plt.subplots(1,4)
    ax[0].imshow(img_init)
    ax[1].imshow(img_gt)
    ax[2].imshow(combine_imgs(img_init, img_gt))
    ax[3].imshow(combine_imgs(img_gt, img_gt))


    plt.show()






