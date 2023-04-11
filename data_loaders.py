import torch
import numpy as np
import trimesh as tm
import albumentations as A
import random
import os
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
from se3_helpers import get_T_CO_init_and_gt
from renderer import render_scene
import threading
import time
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
    data_dir = os.path.join("model3d-datasets",dataset_name, classname, test_or_train)
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

def normalize_depth(depth_img):
    if np.max(depth_img) <= 0.1:
        return depth_img
    mean_val = np.mean(depth_img[depth_img>0.01])
    std = np.std(depth_img[depth_img>0.01])
    if np.isclose(std, 0.0):
        return depth_img
    normalized = np.where(depth_img>0.01, (depth_img-mean_val)/std, 0.0)
    return normalized.astype(np.float32)

def normalize_img(img):
    std = np.std(img)
    mean = np.mean(img)
    return (img-mean)/std

def render_batch(T_COs, mesh_paths, Ks, img_size, parallel_render=False):

    now = time.time()
    if parallel_render:
        imgs, norm_depths = render_batch_parallel(T_COs, mesh_paths, cam_config)
        then = time.time()
        dur = then-now
        print("Par render_time:", dur)
        return imgs, norm_depths
    else:
        imgs, norm_depths = render_batch_sequential(T_COs, mesh_paths, Ks, img_size)
        then = time.time()
        dur = then-now
        print("Seq render_time:", dur)
        return imgs, norm_depths

def render_batch_sequential(T_COs, mesh_paths, Ks, img_size):
    bsz = len(mesh_paths)
    print(img_size)
    assert T_COs.shape == (bsz,4,4)
    assert Ks.shape == (bsz,3,3)
    imgs = []
    norm_depths = []
    for i in range(bsz):
        mesh_path = mesh_paths[i]
        T_CO = T_COs[i]
        K = Ks[i]
        img, norm_depth = render_scene(mesh_path, T_CO, K=K, img_size=img_size)
        if np.max(img)>0:
            img = img/np.max(img)
        imgs.append(img.astype(np.float32))
        norm_depths.append(norm_depth.astype(np.float32))
    imgs = np.array(imgs, dtype=np.float32)
    norm_depths = np.array(norm_depths)
    return imgs, norm_depths

def worker_render_img(i, mesh_path, T_CO, cam_config, imgs_out, depths_out):
    #print("worker", i)
    img, norm_depth = render_scene(mesh_path, T_CO, cam_config)
    imgs_out[i,:,:,:] = img
    depths_out[i,:,:] = norm_depth
    #print("worker", i, "finished")


def render_batch_parallel(T_COs, mesh_paths, cam_config):
    bsz = len(mesh_paths)
    assert T_COs.shape == (bsz,4,4)
    img_res = cam_config["image_resolution"]
    imgs = np.zeros((bsz, img_res, img_res, 3), dtype=np.float32)
    norm_depths = np.zeros((bsz, img_res, img_res), dtype=np.float32)
    threads = []
    for i in range(bsz):
        mesh_path = mesh_paths[i]
        T_CO = T_COs[i]
        t = threading.Thread(target=worker_render_img, args=(i,mesh_path,T_CO,cam_config,imgs,norm_depths))
        t.start()
        threads.append(t)
    for t in threads:
        #print("join", t)
        t.join()

    #print(" ### FINISHED ### ")
    return imgs, norm_depths

def sample_verts_to_batch(mesh_paths, num_verts_to_sample):
    verts_batch = []
    for mesh_path in mesh_paths:
        verts = sample_vertices_as_tensor(mesh_path, num_verts_to_sample)
        verts_batch.append(verts)
    verts_batch = torch.stack(verts_batch)
    return verts_batch



def prepare_model_input(init_imgs, gt_imgs, depths, use_norm_depth=False):
    model_input_batch = []
    for i in range(len(init_imgs)):
        init_img = init_imgs[i]
        #init_img = A.RandomBrightnessContrast()(image=init_img)["image"]
        gt_img = gt_imgs[i]
        gt_img = normalize_img(gt_img)
        #init_img = normalize_img(init_img)
        gt_tensor = transforms(image=gt_img.astype(np.float32))["image"]
        init_tensor = transforms(image=init_img.astype(np.float32))["image"]
        if not use_norm_depth:
            model_input = torch.cat([init_tensor, gt_tensor])
        else:
            norm_depth = normalize_depth(depths[i])
            norm_depth = np.expand_dims(norm_depth, 0)
            norm_depth = torch.tensor(norm_depth)
            model_input = torch.cat([norm_depth, init_tensor, gt_tensor])
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
    import sys
    import time
    sys.path.append("configs")
    from baseline_cfg import get_config
    config = get_config()
    print(config)
    scene_config = config["scene_config"]
    print(scene_config)
    cam_config = config["camera_intrinsics"]

    T_CO_init, T_CO_gt = sample_T_CO_inits_and_gts(128, scene_config)
    mesh_paths = sample_mesh_paths(128, "ModelNet40-norm-ply", ["airplane"], "train")
    
    cur = time.time()
    #render_batch_parallel(T_CO_init, mesh_paths, cam_config)
    render_batch(T_CO_init, mesh_paths, cam_config)
    fin = time.time()
    print(fin-cur)

    """
    cur = time.time()


    fin = time.time()
    print(fin-cur)
    """








