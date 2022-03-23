import torch
import numpy as np
from config import get_config
import matplotlib.pyplot as plt
from data_loaders import prepare_data, get_all_train_or_test_paths, sample_random_path, get_camera_mat_tensor
from se3_helpers import get_T_CO_init_and_gt
from renderer import render_scene
from models import fetch_network
from rotation_representation import calculate_T_CO_pred
import os
import torch

def get_test_batch(batch_size, ds_name, test_classes, scene_config, cam_config):
    T_CO_inits = []
    T_CO_gts = []
    init_imgs = []
    gt_imgs = []
    model_input_batch = []
    mesh_paths = []


    for _ in range(batch_size):
        T_CO_init, T_CO_gt = get_T_CO_init_and_gt(scene_config)
        mesh_path = sample_random_path(ds_name, test_classes, "test")
        init_img = render_scene(mesh_path, T_CO_init.data[0], cam_config)
        gt_img = render_scene(mesh_path, T_CO_gt.data[0], cam_config)
        model_input, T_CO_init_tensor, T_CO_gt_tensor = prepare_data(T_CO_init, T_CO_gt, init_img, gt_img)
        
        T_CO_inits.append(T_CO_init_tensor)
        T_CO_gts.append(T_CO_gt_tensor)
        model_input_batch.append(model_input)
        init_imgs.append(init_img)
        gt_imgs.append(gt_img)
        mesh_paths.append(mesh_path)

    cam_mats = get_camera_mat_tensor(cam_config, batch_size)
    model_input_batch = torch.stack(model_input_batch)
    T_CO_inits = torch.stack(T_CO_inits)
    T_CO_gts = torch.stack(T_CO_gts)
    return model_input_batch, T_CO_inits, T_CO_gts, cam_mats, init_imgs, gt_imgs, mesh_paths

def render_batch(T_COs, mesh_paths, cam_config):
    bsz = len(mesh_paths)
    assert T_COs.shape == (bsz,4,4)

    imgs = []

    for i in range(bsz):
        mesh_path = mesh_paths[i]
        T_CO = T_COs[i]
        img = render_scene(mesh_path, T_CO, cam_config)
        imgs.append(img)
    return imgs

def get_batch_from_prev_prediction(T_CO_preds, T_CO_gts, gt_imgs, batch_size, cam_config, mesh_paths):
    pred_imgs = render_batch(T_CO_preds, mesh_paths, cam_config)
    model_inputs = []
    #T_CO_preds = T_CO_preds
    for i in range(batch_size):
        pred_img = pred_imgs[i]
        T_CO_pred = T_CO_preds[i]
        T_CO_gt = T_CO_gts[i]
        gt_img = gt_imgs[i]
        model_input, T_CO_init, _, prepare_data(T_CO_pred, T_CO_gt, pred_img, gt_img)







def combine_imgs(img1, img2):
    gs1 = np.mean(img1, axis=2)
    gs2 = np.mean(img2, axis=2)
    img = np.zeros((gs1.shape[0], gs1.shape[1], 3))
    img[:,:,0] = gs1
    img[:,:,1] = gs2
    return img



BATCH_SIZE = 4

config = get_config()
scene_config = config["scene_config"]
ds_name = config["train_params"]["dataset_name"]
cam_config = config["camera_intrinsics"]
model_load_dir = config["test_config"]["model_load_dir"]
model_load_name = config["test_config"]["model_load_name"]
model_load_path = os.path.join(model_load_dir, model_load_name)
test_classes = config["test_config"]["test_classes"]
model_name = config["network"]["backend_network"]
rot_repr = config["network"]["rotation_representation"]

model = fetch_network(model_name, rot_repr, use_pretrained=True, pretrained_path=model_load_path)
model.eval()



model_input, T_CO_init, T_CO_gt, cam_mats, init_imgs, gt_imgs, mesh_paths = get_test_batch(
        BATCH_SIZE, ds_name, test_classes, scene_config, cam_config
)

model_output = model(model_input)
T_CO_pred = calculate_T_CO_pred(model_output, T_CO_init, rot_repr, cam_mats)
T_CO_pred = T_CO_pred.detach().cpu().numpy()
pred_imgs = render_batch(T_CO_pred, mesh_paths, cam_config)



fig, ax = plt.subplots(BATCH_SIZE, 2)
for i in range(BATCH_SIZE):
    gt_img = gt_imgs[i]
    init_img = init_imgs[i]
    pred_img = pred_imgs[i]
    ax[i,0].imshow(combine_imgs(init_img, gt_img))
    ax[i,1].imshow(combine_imgs(pred_img, gt_img))

plt.show()

















