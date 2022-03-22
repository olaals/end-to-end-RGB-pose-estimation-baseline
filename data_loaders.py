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


def get_all_training_paths(classes):
    ds_name = "ModelNet40-norm-ply"
    all_training_paths = []
    for classname in classes:
        class_train_paths = get_dataset_class_paths(ds_name, classname, "train")
        all_training_paths = all_training_paths + class_train_paths
    return all_training_paths

def sample_random_training_path(classes):
    all_train_paths = get_all_training_paths(classes)
    random_path = random.choice(all_train_paths)
    return random_path


def prepare_data(T_CO_init, T_CO_gt, init_img, gt_img):
    gt_tensor = transforms(image=gt_img.astype(np.float32))["image"]
    init_tensor = transforms(image=init_img.astype(np.float32))["image"]
    T_CO_gt_tensor = torch.from_numpy(T_CO_gt.data[0].astype(np.float32))
    T_CO_init_tensor = torch.from_numpy(T_CO_init.data[0].astype(np.float32))
    model_input = torch.cat([init_tensor, gt_tensor])
    return model_input, T_CO_init_tensor, T_CO_gt_tensor

def get_training_example(config):
    train_classes = config["train_params"]["train_classes"]
    num_sample_verts = config["train_params"]["num_sample_vertices"]


    T_CO_init, T_CO_gt = get_T_CO_init_and_gt(config)
    mesh_path = sample_random_training_path(train_classes)
    init_img = render_scene(mesh_path, T_CO_init.data[0], config)
    gt_img = render_scene(mesh_path, T_CO_gt.data[0], config)
    model_input, T_CO_init_tensor, T_CO_gt_tensor = prepare_data(T_CO_init, T_CO_gt, init_img, gt_img)
    mesh_vertices = sample_vertices_as_tensor(mesh_path, num_sample_verts)
    return model_input, T_CO_init_tensor, T_CO_gt_tensor, mesh_vertices

def get_train_batch(config):
    batch_size = config["train_params"]["batch_size"]

    model_input_batch = []
    T_CO_init_batch = []
    T_CO_gt_batch = []
    mesh_vert_batch = []
    for _ in range(batch_size):
        model_input, T_CO_init, T_CO_gt, mesh_verts = get_training_example(config)
        model_input_batch.append(model_input)
        T_CO_init_batch.append(T_CO_init)
        T_CO_gt_batch.append(T_CO_gt)
        mesh_vert_batch.append(mesh_verts)
    model_input_batch = torch.stack(model_input_batch)
    T_CO_init_batch = torch.stack(T_CO_init_batch)
    T_CO_gt_batch = torch.stack(T_CO_gt_batch)
    mesh_vert_batch = torch.stack(mesh_vert_batch)
    return model_input_batch, T_CO_init_batch, T_CO_gt_batch, mesh_vert_batch


if __name__ == '__main__':
    config = get_config()
    model_input, T_CO_init, T_CO_gt, mesh_verts = get_train_batch(config)
    print("Model input", model_input.shape)
    print("T_CO_init", T_CO_init.shape)
    print("T_CO_gt", T_CO_gt.shape)
    print("mesh_verts", mesh_verts.shape)







