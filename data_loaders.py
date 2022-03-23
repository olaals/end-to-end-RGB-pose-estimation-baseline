import torch
import numpy as np
import trimesh as tm
import random
import os
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
from se3_helpers import get_T_CO_init_and_gt
from renderer import render_scene
from config import get_config

transforms = A.Compose([
    ToTensorV2()
])


def get_vertices(mesh_path):
    mesh = tm.load(mesh_path)
    vertices = mesh.vertices
    return vertices

def sample_vertices_as_tensor(mesh_path, num_verts=1000):
    verts = get_vertices(mesh_path)
    sampled_verts = []
    for i in range(num_verts):
        vert = np.array(random.choice(verts))
        sampled_verts.append(vert)
    sampled_verts = np.array(sampled_verts, dtype=np.float32)
    tensor_verts = torch.tensor(sampled_verts)
    return tensor_verts

def get_dataset_class_paths(dataset_name, classname, test_or_train):
    data_dir = os.path.join(dataset_name, classname, test_or_train)
    model_paths = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
    return model_paths


def get_all_train_or_test_paths(ds_name, classes, train_or_test):
    all_training_paths = []
    for classname in classes:
        class_train_paths = get_dataset_class_paths(ds_name, classname, train_or_test)
        all_training_paths = all_training_paths + class_train_paths
    return all_training_paths

def sample_random_path(ds_name,classes, train_or_test):
    all_train_paths = get_all_train_or_test_paths(ds_name,classes, train_or_test)
    random_path = random.choice(all_train_paths)
    return random_path

def sample_mesh_paths(batch_size, ds_name, classes, train_or_test):
    mesh_paths = []
    for _ in range(batch_size):
        mesh_paths.append(sample_random_path(ds_name, classes, train_or_test))
    return mesh_paths

def sample_T_CO_inits_and_gts(batch_size, scene_config):
    T_CO_inits = []
    T_CO_gts = []
    for _ in range(batch_size):
        T_CO_init, T_CO_gt = get_T_CO_init_and_gt(scene_config)
        T_CO_inits.append(T_CO_init.data[0].astype(np.float32))
        T_CO_gts.append(T_CO_gt.data[0].astype(np.float32))
    T_CO_inits = np.array(T_CO_inits, dtype=np.float32)
    T_CO_gts = np.array(T_CO_gts, dtype=np.float32)
    return T_CO_inits, T_CO_gts


def render_batch(T_COs, mesh_paths, cam_config):
    bsz = len(mesh_paths)
    assert T_COs.shape == (bsz,4,4)
    imgs = []
    for i in range(bsz):
        mesh_path = mesh_paths[i]
        T_CO = T_COs[i]
        img = render_scene(mesh_path, T_CO, cam_config)
        imgs.append(img.astype(np.float32))
    imgs = np.array(imgs, dtype=np.float32)
    return imgs

def sample_verts_to_batch(mesh_paths, num_verts_to_sample):
    verts_batch = []
    for mesh_path in mesh_paths:
        verts = sample_vertices_as_tensor(mesh_path, num_verts_to_sample)
        verts_batch.append(verts)
    verts_batch = torch.stack(verts_batch)
    return verts_batch

def prepare_model_input(init_imgs, gt_imgs):
    model_input_batch = []
    for init_img, gt_img in zip(init_imgs, gt_imgs):
        gt_tensor = transforms(image=gt_img.astype(np.float32))["image"]
        init_tensor = transforms(image=init_img.astype(np.float32))["image"]
        model_input = torch.cat([init_tensor, gt_tensor])
        model_input_batch.append(model_input)
    model_input_batch = torch.stack(model_input_batch)
    return model_input_batch

        

def get_camera_mat_tensor(cam_intrinsics, batch_size):
    sensor_width = cam_intrinsics["sensor_width"]
    focal_len = cam_intrinsics["focal_length"]
    img_res = cam_intrinsics["image_resolution"]
    px_size = sensor_width/img_res
    fx = fy = focal_len/px_size
    u = v = img_res/2
    K = np.array([[fx, 0.0, u], [0.0, fy, v], [0.0,0.0,1.0]], dtype=np.float32)
    K_tens = torch.tensor(K)
    K_tens_batch = K_tens.unsqueeze(0).repeat(batch_size, 1,1)
    return K_tens_batch





if __name__ == '__main__':
    cam_intr = {
        "focal_length": 50, #mm
        "sensor_width": 36, #mm
        "image_resolution": 300, # width=height
    }
    bsz = 8
    Ks = get_camera_mat_tensor(cam_intr, bsz)
    print(Ks)
    


    pass







