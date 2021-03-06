import torch
from data_loaders import *
from image_dataloaders import get_dataloaders
from loss import compute_ADD_L1_loss, compute_disentangled_ADD_L1_loss
from rotation_representation import calculate_T_CO_pred
#from models.efficient_net import 
from models import fetch_network
import os
from parser_config import get_dict_from_cli
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sb


class MeshPathDataset(Dataset):
    def __init__(self, mesh_paths):
        self.mesh_paths = mesh_paths
    def __len__(self):
        return len(self.mesh_paths)
    def __getitem__(self, idx):
        return self.mesh_paths[idx]




def save_loss_bar_plot(loss_dict, logdir):
    print(len(loss_dict))
    sb_dict = {}
    sb_dict["x"] = []
    sb_dict["y"] = []
    sb_dict["group"] = []
    for test_class in loss_dict:
        loss_array = loss_dict[test_class]
        average_loss_per_pred_iter = np.mean(np.mean(loss_array, axis=0), axis=0)
        print(average_loss_per_pred_iter)
        for i in range(len(average_loss_per_pred_iter)):
            sb_dict["x"].append(test_class)
            sb_dict["y"].append(average_loss_per_pred_iter[i])
            sb_dict["group"].append(f'pred iter {i}')
    sb.barplot(x='x', y='y', hue="group", data=sb_dict)
    save_path = os.path.join(logdir, "barplot.png")
    plt.savefig(save_path)



def validate_model(model, config, val_or_test):
    print("Validating Model")
    model.eval()
    scene_config = config["scene_config"]
    use_norm_depth = config["advanced"]["use_normalized_depth"]
    rotation_repr = config["network"]["rotation_representation"]

    ds_name = config["dataset_config"]["model3d_dataset"]
    classes = config["test_config"]["test_classes"]
    img_size = config["camera_intrinsics"]["image_resolution"]


    cam_intrinsics = config["camera_intrinsics"]
    device = config["train_params"]["device"]
    model = model.to(device)

    #test params
    batch_size = config["test_config"]["batch_size"]
    num_train_batches = config["train_params"]["num_batches_to_train"]
    num_sample_verts = config["train_params"]["num_sample_vertices"]
    device = config["train_params"]["device"]
    use_par_render = config["scene_config"]["use_parallel_rendering"]

    test_iterations_per_class= config["test_config"]["iterations_per_class"]
    test_predict_iterations = config["test_config"]["predict_iterations"]

    ds_conf = config["dataset_config"]
    _,val_loader,test_loader = get_dataloaders(ds_conf, batch_size)
    if val_or_test == 'val':
        data_loader = val_loader
    elif val_or_test =='test':
        data_loader = test_loader
    print("Batch size", batch_size)
    print(f'Validating on device', device)
    print("on classes \n", classes)
    results = np.zeros((test_predict_iterations, len(data_loader.dataset)))
    examples = 0
    with torch.no_grad():
        for i, (init_imgs, gt_imgs, T_CO_init, T_CO_gt, mesh_verts, mesh_paths, depths, cam_mats) in enumerate(data_loader):
            depths = depths.numpy()
            bsz = len(T_CO_init)
            #cam_mats = get_camera_mat_tensor(cam_intrinsics, bsz).to(device)
            T_CO_pred = T_CO_init
            gt_imgs = gt_imgs.numpy()
            mesh_verts = mesh_verts.to(device)
            T_CO_gt = T_CO_gt.to(device)

            for j in range(test_predict_iterations):
                if(j==0):
                    pred_imgs = init_imgs.numpy()
                    T_CO_pred = T_CO_pred.to(device)
                else:
                    pred_imgs, depths = render_batch(T_CO_pred, mesh_paths, cam_mats.numpy(), img_size, use_par_render)
                    T_CO_pred = torch.tensor(T_CO_pred).to(device)
                model_input = prepare_model_input(pred_imgs, gt_imgs, depths, use_norm_depth).to(device)
                model_output = model(model_input)
                T_CO_pred_new = calculate_T_CO_pred(model_output, T_CO_pred, rotation_repr, cam_mats)
                addl1_loss = compute_ADD_L1_loss(T_CO_gt, T_CO_pred_new, mesh_verts, use_batch_mean=False)
                addl1_loss = addl1_loss.detach().cpu().numpy()
                T_CO_pred = T_CO_pred_new.detach().cpu().numpy()
                results[j][examples:(examples+bsz)] = addl1_loss
            examples += bsz
    return np.mean(results,axis=1)


        




def evaluate_model(model, config, test_or_train_ds, use_all_examples=True, max_examples_from_each_class=10):
    model.eval()
    scene_config = config["scene_config"]
    use_norm_depth = config["advanced"]["use_normalized_depth"]
    rotation_repr = config["network"]["rotation_representation"]

    ds_name = config["dataset_config"]["model3d_dataset"]
    if(test_or_train_ds == 'train'):
        classes = config["dataset_config"]["train_classes"]
    elif(test_or_train_ds == 'test'):
        classes = config["test_config"]["test_classes"]
    else:
        assert False


    cam_intrinsics = config["camera_intrinsics"]

    device = config["train_params"]["device"]
    model = model.to(device)

    #test params
    batch_size = config["test_config"]["batch_size"]
    num_train_batches = config["train_params"]["num_batches_to_train"]
    num_sample_verts = config["train_params"]["num_sample_vertices"]
    device = config["train_params"]["device"]
    use_par_render = config["scene_config"]["use_parallel_rendering"]

    test_iterations_per_class= config["test_config"]["iterations_per_class"]
    test_predict_iterations = config["test_config"]["predict_iterations"]

    print("Batch size", batch_size)
    print("Testing on device", device)
    print("Testing on classes \n", classes)

    loss_dict = {}

    for test_class in classes:
        all_class_test_paths = get_dataset_class_paths(ds_name, test_class, test_or_train_ds)
        if not use_all_examples:
            if len(all_class_test_paths) > max_examples_from_each_class:
                all_class_test_paths = all_class_test_paths[:max_examples_from_each_class]
        loss_dict[test_class] = np.zeros((len(all_class_test_paths), test_iterations_per_class, test_predict_iterations))
        for test_iter in range(test_iterations_per_class):
            seen_train_ex = 0
            print(f'Test iteration {test_iter} on class {test_class}')
            mesh_path_loader = DataLoader(MeshPathDataset(all_class_test_paths), batch_size=batch_size, shuffle=False)
            for mesh_paths in mesh_path_loader:
                actual_bsz = len(mesh_paths) # update batch_size since last batch in dataset may have different size
                with torch.no_grad():
                    T_CO_init, T_CO_gt = sample_T_CO_inits_and_gts(actual_bsz, scene_config)
                    mesh_verts = sample_verts_to_batch(mesh_paths, num_sample_verts).to(device)
                    cam_mats = get_camera_mat_tensor(cam_intrinsics, actual_bsz).to(device)
                    gt_imgs,_ = render_batch(T_CO_gt, mesh_paths, cam_intrinsics, use_par_render)
                    T_CO_gt = torch.tensor(T_CO_gt).to(device)
                    T_CO_pred = T_CO_init # current pred is initial
                    seen_train_ex += actual_bsz
                    for pred_iter in range(test_predict_iterations):
                        pred_imgs, norm_depth = render_batch(T_CO_pred, mesh_paths, cam_intrinsics, use_par_render)
                        model_input = prepare_model_input(pred_imgs, gt_imgs, norm_depth, use_norm_depth).to(device)
                        T_CO_pred = torch.tensor(T_CO_pred).to(device)
                        model_output = model(model_input)
                        T_CO_pred_new = calculate_T_CO_pred(model_output, T_CO_pred, rotation_repr, cam_mats)
                        T_CO_pred = T_CO_pred_new.detach().cpu().numpy()
                        loss = compute_ADD_L1_loss(T_CO_gt, T_CO_pred_new, mesh_verts, use_batch_mean = False)
                        #print(f'Loss for class {test_class}, pred iter {pred_iter}:', loss.mean().item())
                        #print(loss.cpu().numpy())
                        #print(f'The current actual batch size is {actual_bsz}')
                        print("seen_train_ex:", seen_train_ex, "bsz", actual_bsz, "test_iter", test_iter, "pred_iter", pred_iter)
                        loss_dict[test_class][seen_train_ex-actual_bsz:seen_train_ex, test_iter, pred_iter] = loss.cpu().numpy()


    


    num_classes = 0
    mean_losses = np.zeros((test_predict_iterations))
    out_dict = {}
    for classname in loss_dict:
        loss_array = loss_dict[classname]
        average_loss_per_pred_iter = np.mean(np.mean(loss_array, axis=0), axis=0)
        out_dict[classname] = {}
        num_classes += 1
        for i in range(len(average_loss_per_pred_iter)):
            loss = average_loss_per_pred_iter[i]
            out_dict[classname]["iter"+str(i)] = loss
            mean_losses[i] += loss

    mean_losses = mean_losses/num_classes

        
    


    return out_dict, mean_losses

    



if __name__ == '__main__':
    try:
        config = get_dict_from_cli()
        # model load parameters
        model_name = config["network"]["backend_network"]
        rotation_repr = config["network"]["rotation_representation"]
        use_pretrained = config["model_io"]["use_pretrained_model"]
        model_save_dir = config["model_io"]["model_save_dir"]
        pretrained_name = config["model_io"]["pretrained_model_name"]
        pretrained_path = os.path.join(model_save_dir, pretrained_name)
        use_norm_depth = config["advanced"]["use_normalized_depth"]
        model = fetch_network(model_name, rotation_repr, use_norm_depth, use_pretrained, pretrained_path)
        logdir = config["logging"]["logdir"]
        os.makedirs(logdir, exist_ok=True)
        # print training info
        print("")
        print(" ### TESTING IS STARTING ### ")
        print("Loading backend network", model_name.upper(), "with rotation representation", rotation_repr)
        mean_loss = validate_model(model, config, "test")
        print("Mean loss", mean_loss)
        #save_loss_bar_plot(loss_dict, logdir)
    except:
        raise Exception("Include a valid config file with: ".upper()+"python train_model.py baseline_cfg")



