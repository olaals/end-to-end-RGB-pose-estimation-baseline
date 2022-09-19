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
from image_dataloaders import get_dataloader
from PIL import Image
from scipy.signal import convolve2d
import cv2


def create_silhouette(img, silhouette_only=False):
    gray = np.mean(img, axis=2)
    sil = np.where(gray>0, 1.0, 0.0)
    filt = np.ones((5,5))
    res = convolve2d(sil, filt, mode='same')
    res = np.clip(res, 0.0, 1.0)
    sil = res-sil
    if silhouette_only:
        return sil
    else:
        img[sil>0] = (1.0,0.0, 0.0)
        return img


def blend_imgs(im1, im2, alpha):
    im1 = Image.fromarray(np.uint8(im1*255.0))
    im2 = Image.fromarray(np.uint8(im2*255.0))
    return np.asarray(Image.blend(im1,im2,alpha))/255.0

def save_img(img, resize_to, save_dir, filename):
    save_path = os.path.join(save_dir, filename)
    img_cv2 = cv2.resize(cv2.cvtColor(np.uint8(img*255), cv2.COLOR_RGB2BGR), resize_to)
    cv2.imwrite(save_path, img_cv2)



def create_interpolated_image_fig(init_imgs, gt_imgs, pred_imgs_sequence, gt_imgs_raster=None,  save_dir=None, show_fig=False, train_examples="", train_val_or_test =None):
    print("save_dir:", save_dir)
    BLEND_ALPHA = 0.4
    IMG_SIZE = (200,200)

    save_figure = (save_dir is not None)

    batch_size = len(init_imgs)
    iter_num = len(pred_imgs_sequence)
    fig, ax = plt.subplots(batch_size, iter_num+2)
    fig.set_size_inches(18.5, 10.5, forward=True)
    if(save_dir is not None):
        assert(train_val_or_test is not None)
        save_subdir = os.path.join(save_dir, train_val_or_test)
        print("save subdir", save_subdir)
        os.makedirs(save_subdir, exist_ok=True)
        save_path = os.path.join(save_subdir, "viz_at_train_ex"+str(train_examples)+".png")
        print(save_path)
        save_full_img_subdir = os.path.join(save_subdir, "train_ex"+str(train_examples))
        os.makedirs(save_full_img_subdir, exist_ok=True)
        
    #ax = fig.add_subplot(batch_size, iter_num+1)
    for i in range(batch_size):
        init_img = init_imgs[i]
        gt_img = gt_imgs[i]
        ax[i,0].imshow(gt_img)
        ax[i,0].axis('off')
        ax[i,0].set_title("Real img")
        if(gt_imgs_raster is not None):
            gt_raster = gt_imgs_raster[i]
            sil = create_silhouette(gt_raster, silhouette_only=True)
            gt_img[sil>0] = (0.0,1.0, 0.0)
        initial = blend_imgs(create_silhouette(init_img), gt_img, BLEND_ALPHA)
        
        if save_figure:
            save_full_img_subdir_bsz = os.path.join(save_full_img_subdir, "example"+str(i))
            os.makedirs(save_full_img_subdir_bsz, exist_ok=True)
            save_img(gt_img, IMG_SIZE, save_full_img_subdir_bsz, "0-gt.png")
            save_img(initial, IMG_SIZE, save_full_img_subdir_bsz, "1-init.png")
        ax[i,1].imshow(initial)
        ax[i,1].axis('off')
        ax[i,1].set_title("Green: GT. Red: init")
        for j in range(iter_num):
            gt_img = gt_imgs[i]
            init_img = init_imgs[i]
            pred_img = pred_imgs_sequence[j][i]
            blend_pred = blend_imgs(create_silhouette(pred_img), gt_img, BLEND_ALPHA)
            if save_figure:
                save_img(blend_pred, IMG_SIZE, save_full_img_subdir_bsz, f'{j+2}-pred-{j}.png')
            ax[i,j+2].imshow(blend_pred)
            ax[i,j+2].axis('off')
            ax[i,j+2].set_title(f'Green: GT. Red: pred {j}')

    if show_fig:
        plt.show()
    if(save_figure):
        print("Saving visualization:", save_path)
        plt.savefig(save_path,dpi=150, bbox_inches='tight')


def combine_imgs(img1, img2):
    gs1 = np.mean(img1, axis=2)
    gs2 = np.mean(img2, axis=2)
    img = np.zeros((gs1.shape[0], gs1.shape[1], 3))
    img[:,:,0] = gs1
    img[:,:,1] = gs2
    return img

def create_rgb_overlapped_image_fig(init_imgs, gt_imgs, pred_imgs_sequence, save_dir, train_val_or_test, show_fig, save_fig):
    print("save_dir:", save_dir)
    batch_size = len(init_imgs)
    iter_num = len(pred_imgs_sequence)
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
        plt.savefig(save_path,dpi=300, bbox_inches='tight')

def visualize_examples(model, config, train_val_or_test, show_fig=False, save_dir=None, n_train_examples=""):

    assert (train_val_or_test=="test" or train_val_or_test=="train" or train_val_or_test=='val')

    batch_size = 5
    device = config["train_params"]["device"]
    iter_num = 5

    scene_config = config["scene_config"]
    ds_name = config["dataset_config"]["model3d_dataset"]
    img_size= config["camera_intrinsics"]["image_resolution"]
    print("img size", img_size)
    rot_repr = config["network"]["rotation_representation"]
    use_norm_depth = config["advanced"]["use_normalized_depth"]
    use_par_render = config["scene_config"]["use_parallel_rendering"]
    if(train_val_or_test == 'train' or train_val_or_test=='val'):
        ds_conf = config["dataset_config"]
    elif(train_val_or_test=='test'):
        ds_conf = config["test_dataset_config"]

    #model = fetch_network(model_name, rot_repr, use_norm_depth, use_pretrained=True, pretrained_path=model_load_path)
    model.eval()
    #model = model.to(device)

    train_from_imgs = config["dataset_config"]["train_from_images"]

    data_loader = get_dataloader(ds_conf, batch_size, train_val_or_test)
    init_imgs, gt_imgs, T_CO_init, T_CO_gt, mesh_verts, mesh_paths, depths, cam_mats = next(iter(data_loader))
    T_CO_init = T_CO_init.numpy()
    T_CO_gt = T_CO_gt.numpy()
    gt_imgs = gt_imgs.numpy()
    init_imgs = init_imgs.numpy()
    print(cam_mats.numpy())
    depths = depths.numpy()
    gt_imgs_raster, _ = render_batch(T_CO_gt, mesh_paths, cam_mats.numpy(), img_size, use_par_render)

    T_CO_pred = T_CO_init
    pred_imgs = init_imgs
    pred_imgs_sequence = []
    T_CO_gt = torch.tensor(T_CO_gt).to(device)
    with torch.no_grad():
        for i in range(iter_num):
            model_input = prepare_model_input(pred_imgs, gt_imgs, depths, use_norm_depth).to(device)
            #cam_mats = get_camera_mat_tensor(cam_intrinsics, batch_size).to(device)
            #mesh_verts = sample_verts_to_batch(mesh_paths, num_sample_verts).to(device)

            T_CO_pred = torch.tensor(T_CO_pred).to(device)
            model_output = model(model_input)
            T_CO_pred = calculate_T_CO_pred(model_output, T_CO_pred, rot_repr, cam_mats)
            T_CO_pred = T_CO_pred.detach().cpu().numpy()
            pred_imgs, depths = render_batch(T_CO_pred, mesh_paths, cam_mats.numpy(), img_size, use_par_render)
            if not use_norm_depth: norm_depth=None
            
            pred_imgs_sequence.append(pred_imgs)

    #create_rgb_overlapped_image_fig(init_imgs, gt_imgs, pred_imgs_sequence, save_path, show_fig, save_fig)
    print("Creating images")
    create_interpolated_image_fig(init_imgs, gt_imgs, pred_imgs_sequence, gt_imgs_raster, save_dir, show_fig, n_train_examples, train_val_or_test)



if __name__ == '__main__':
    config = get_dict_from_cli()
    model_name = config["network"]["backend_network"]
    rotation_repr = config["network"]["rotation_representation"]
    use_pretrained = config["model_io"]["use_pretrained_model"]
    model_save_dir = config["model_io"]["model_save_dir"]
    model_save_name = config["model_io"]["model_save_name"]
    train_ds = config["dataset_config"]["img_dataset"]
    pretrained_path = os.path.join(model_save_dir, model_save_name)
    print("Loading pretrained from", pretrained_path)
    use_norm_depth = config["advanced"]["use_normalized_depth"]
    model = fetch_network(model_name, rotation_repr, use_norm_depth, True, pretrained_path).to('cuda')
    visualize_examples(model, config, "test", show_fig=True)
        









