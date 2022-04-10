import torch
import numpy as np
import matplotlib.pyplot as plt
from se3_helpers import get_T_CO_init_and_gt
from renderer import render_scene
from models import fetch_network
from rotation_representation import calculate_T_CO_pred
import os
import torch
from data_loaders import *
from parser_config import get_dict_from_cli
from loss import compute_ADD_L1_loss





def combine_imgs(img1, img2):
    gs1 = np.mean(img1, axis=2)
    gs2 = np.mean(img2, axis=2)
    img = np.zeros((gs1.shape[0], gs1.shape[1], 3))
    img[:,:,0] = gs1
    img[:,:,1] = gs2
    return img


def visualize_examples(config, train_or_test, show_fig=False, save_fig=False, save_path=""):
    assert (train_or_test=="test" or train_or_test=="train")

    batch_size = 5
    device = 'cpu'
    iter_num = 5

    scene_config = config["scene_config"]
    ds_name = config["train_params"]["dataset_name"]
    cam_intrinsics = config["camera_intrinsics"]
    model_load_dir = config["test_config"]["model_load_dir"]
    model_load_name = config["test_config"]["model_load_name"]
    model_load_path = os.path.join(model_load_dir, model_load_name)
    test_classes = config["test_config"]["test_classes"]
    model_name = config["network"]["backend_network"]
    rot_repr = config["network"]["rotation_representation"]
    use_norm_depth = config["advanced"]["use_normalized_depth"]
    use_par_render = config["scene_config"]["use_parallel_rendering"]

    if train_or_test == "train":
        classes = config["train_params"]["train_classes"]
    else:
        classes = config["test_config"]["test_classes"]


        



    print("Loading pretrained network", model_load_name)
    print("Visualization for classes", classes, "from dataset", train_or_test)
    model = fetch_network(model_name, rot_repr, use_norm_depth, use_pretrained=True, pretrained_path=model_load_path)
    # Beware: model.eval() behaves weird with efficientnet in certain situations, not figured out why, try commenting it out
    model.eval()


    T_CO_init, T_CO_gt = sample_T_CO_inits_and_gts(batch_size, scene_config)
    mesh_paths = sample_mesh_paths(batch_size, ds_name, classes, train_or_test)
    print(mesh_paths)
    init_imgs, norm_depth = render_batch(T_CO_init, mesh_paths, cam_intrinsics, use_par_render)
    #if not use_norm_depth: norm_depth=None
    gt_imgs, _ = render_batch(T_CO_gt, mesh_paths, cam_intrinsics, use_par_render)

    T_CO_pred = T_CO_init
    pred_imgs = init_imgs
    pred_imgs_sequence = []
    T_CO_gt = torch.tensor(T_CO_gt).to(device)
    with torch.no_grad():
        for i in range(iter_num):
            model_input = prepare_model_input(pred_imgs, gt_imgs, norm_depth, use_norm_depth).to(device)
            cam_mats = get_camera_mat_tensor(cam_intrinsics, batch_size).to(device)
            #mesh_verts = sample_verts_to_batch(mesh_paths, num_sample_verts).to(device)

            T_CO_pred = torch.tensor(T_CO_pred).to(device)
            model_output = model(model_input)
            T_CO_pred = calculate_T_CO_pred(model_output, T_CO_pred, rot_repr, cam_mats)
            T_CO_pred = T_CO_pred.detach().cpu().numpy()
            pred_imgs, norm_depth = render_batch(T_CO_pred, mesh_paths, cam_intrinsics, use_par_render)
            if not use_norm_depth: norm_depth=None
            
            pred_imgs_sequence.append(pred_imgs)






    fig, ax = plt.subplots(batch_size, iter_num+1)
    fig.set_size_inches(18.5, 10.5, forward=True)
    #ax = fig.add_subplot(batch_size, iter_num+1)
    for i in range(batch_size):
        init_img = init_imgs[i]
        gt_img = gt_imgs[i]
        ax[i,0].imshow(combine_imgs(init_img, gt_img))
        ax[i,0].axis('off')
        ax[i,0].set_title("Green: GT. Red: init")
        for j in range(iter_num):
            gt_img = gt_imgs[i]
            init_img = init_imgs[i]
            pred_img = pred_imgs_sequence[j][i]
            ax[i,j+1].imshow(combine_imgs(pred_img, gt_img))
            ax[i,j+1].axis('off')
            ax[i,j+1].set_title(f'Green: GT. Red: pred {j}')

    if show_fig:
        plt.show()
    if save_fig:
        plt.savefig(save_path)







if __name__ == '__main__':
    config = get_dict_from_cli()
    visualize_examples(config, "test", show_fig=True)









