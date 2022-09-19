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
from pyrender.constants import RenderFlags
from spatialmath.base import trnorm


def get_camera_matrix(intrinsics):
    focal_len = intrinsics["focal_length"]
    img_res = intrinsics["image_resolution"]
    sensor_width = intrinsics["sensor_width"]
    pix_per_mm = sensor_width/img_res
    fx = fy = focal_len/pix_per_mm
    vx = vy = img_res/2
    K = np.array([[fx, 0, vx],[0, fy, vy],[0,0,1]])
    return K


def add_object(scene, path, pose=None, force_mesh=False):
    if(pose is None):
        pose = sm.SE3.Rx(0).data[0]
    if force_mesh:
        trimesh_mesh = tm.load(path, force='mesh')
    else:
        trimesh_mesh = tm.load(path)
    #mat = pyrender.Material(doubleSided=True)
    if(isinstance(trimesh_mesh, tm.scene.scene.Scene)):
        for m in list(trimesh_mesh.geometry.values()):
            if(os.path.splitext(path)[1] == '.glb'): # correction for glb export from blender
                rx = sm.SE3.Rx(90, unit='deg')
                m = m.apply_transform(rx.data[0])
            mesh = pyrender.Mesh.from_trimesh(m, smooth=False)
            scene.add(mesh)
    else:
        if(os.path.splitext(path)[1] == '.glb'): # correction for glb export from blender
            rx = sm.SE3.Rx(90, unit='deg')
            trimesh_mesh = trimesh_mesh.apply_transform(rx.data[0])
        mesh = pyrender.Mesh.from_trimesh(trimesh_mesh, smooth=False)
        scene.add(mesh, pose=pose)

def add_light(scene, T_CO):
    assert T_CO.shape == (4,4)
    T_OC = np.linalg.inv(T_CO)
    light = pyrender.SpotLight(color=np.ones(3), intensity=15.0,
                            innerConeAngle=np.pi/9.0,
                            outerConeAngle=np.pi/2.0)
    scene.add(light, pose=T_OC)

def plot_SE3(T):
    T = trnorm(T)
    R = sm.SO3(trnorm(T[:3,:3]))
    T = sm.SE3.Rt(R, T[:3,3])
    T_orig = sm.SE3.Rx(0)
    T_orig.plot(color='red')
    T.plot( dims=[-3, 3, -3, 3, -3, 3])
    plt.show()

def add_camera(scene, T_CO, K):
    assert T_CO.shape == (4,4)
    T_OC = np.linalg.inv(T_CO)
    #plot_SE3(T_OC)
    fx,fy, ux,uy = K[0,0], K[1,1], K[0,2], K[1,2]
    camera = pyrender.IntrinsicsCamera(fx, fy, ux,uy)
    scene.add(camera, pose=T_OC)

def render(scene, img_size):
    r = pyrender.OffscreenRenderer(img_size, img_size)
    #render_flags = RenderFlags.FACE_NORMALS
    color, depth = r.render(scene)
    r.delete()
    return color/255.0, depth


def render_scene(object_path, T_CO, cam_config=None, K=None, img_size=None):
    assert T_CO.shape == (4,4)
    if (K is not None and img_size is not None):
        K = K
        img_size = img_size
    else:
        assert (cam_config is not None)
        K = get_camera_matrix(cam_config)
        img_size = cam_config["image_resolution"]


    T_CO = sm.SE3.Rx(180, unit='deg').data[0]@T_CO # convert from OpenCV camera frame to OpenGL camera frame
    scene = pyrender.Scene(ambient_light=[0.1,0.1,0.1])
    scene.bg_color = (0,0,0)
    add_object(scene, object_path)
    add_light(scene, T_CO)
    add_camera(scene, T_CO, K)
    img, depth = render(scene, img_size)
    return img, depth


class CustomShaderCache():
    def __init__(self):
        self.program = None

    def get_program(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
        if self.program is None:
            mesh_vert = os.path.join("assets","shaders","mesh.vert")
            mesh_frag = os.path.join("assets","shaders","mesh.frag")
            self.program = pyrender.shader_program.ShaderProgram(mesh_vert, mesh_frag, defines=defines)
        return self.program

def render_normals(object_path, T_CO, cam_config):
    assert T_CO.shape == (4,4)
    img_size = cam_config["image_resolution"]
    K = get_camera_matrix(cam_config)

    T_CO = sm.SE3.Rx(180, unit='deg').data[0]@T_CO # convert from OpenCV camera frame to OpenGL camera frame
    T_identity = sm.SE3.Rx(0).data[0]
    scene = pyrender.Scene()
    scene.bg_color = (0,0,0)
    add_object(scene, object_path, pose=T_CO, force_mesh=True)
    add_camera(scene, T_identity, K)
    renderer = pyrender.OffscreenRenderer(img_size, img_size)
    renderer._renderer._program_cache = CustomShaderCache()
    normals, depth = renderer.render(scene)
    normals = normals/255.0
    return normals


