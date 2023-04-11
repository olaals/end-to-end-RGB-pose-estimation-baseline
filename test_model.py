import torch
from data_loaders import *
from image_dataloaders import get_dataloaders
from loss import compute_ADD_L1_loss, compute_disentangled_ADD_L1_loss, compute_ADD_L2_loss, compute_angular_dist,compute_transl_dist
from rotation_representation import calculate_T_CO_pred
#from models.efficient_net import 
from models import fetch_network
import os
from parser_config import get_dict_from_cli
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sb
from visualization import create_interpolated_image_fig
import json
import pickle
import time


def write_json(file_name, data):
    with open(file_name, 'w') as json_file:
        json.dump(data, json_file, indent=4, sort_keys=True)

def write_pickle(file_name, data):
    with open(file_name, 'wb') as pickle_file:
        pickle.dump(data, pickle_file)



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

    ds_name = config["val_dataset_config"]["model3d_dataset"]
    classes = config["val_dataset_config"]["classes"]
    img_size = config["camera_intrinsics"]["image_resolution"]


    device = config["train_params"]["device"]
    model = model.to(device)

    #test params
    batch_size = config["val_config"]["batch_size"]
    num_train_batches = config["train_params"]["num_batches_to_train"]
    num_sample_verts = config["train_params"]["num_sample_vertices"]
    device = config["train_params"]["device"]
    use_par_render = config["scene_config"]["use_parallel_rendering"]

    test_iterations_per_class= config["val_config"]["iterations_per_class"]
    test_predict_iterations = config["val_config"]["predict_iterations"]

    ds_conf = config["val_dataset_config"]
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

def test_model(model, config):
    print("Testing Model")
    model.eval()
    scene_config = config["scene_config"]
    use_norm_depth = config["advanced"]["use_normalized_depth"]
    rotation_repr = config["network"]["rotation_representation"]

    ds_name = config["test_dataset_config"]["model3d_dataset"]
    classes = config["test_dataset_config"]["classes"]
    img_size = config["camera_intrinsics"]["image_resolution"]


    device = config["train_params"]["device"]
    model = model.to(device)

    #test params
    batch_size = 1
    num_train_batches = config["train_params"]["num_batches_to_train"]
    num_sample_verts = config["train_params"]["num_sample_vertices"]
    device = config["train_params"]["device"]
    use_par_render = config["scene_config"]["use_parallel_rendering"]
    test_iterations_per_class= config["test_config"]["iterations_per_class"]
    test_predict_iterations = config["test_config"]["predict_iterations"]
    print("Test predict iterations")
    print(test_predict_iterations)
    ds_conf = config["test_dataset_config"]

    logdir = config["logging"]["logdir"]
    test_logdir = os.path.join(logdir, "test", config["test_dataset_config"]["img_dataset"])
    os.makedirs(test_logdir, exist_ok=True)
    mean_examples = {}
    print("Batch size", batch_size)
    print(f'Testing on device', device)
    print("on classes \n", classes)
    result_dict = {}
    result_dict_all = {}
    iteration_timings = []
    for test_class in classes:
        test_class_logdir = os.path.join(test_logdir,test_class)
        ds_conf["classes"] = [test_class]
        _,_,data_loader = get_dataloaders(ds_conf, batch_size)
        results = np.zeros((test_predict_iterations, len(data_loader.dataset)))
        examples = 0
        result_dict[test_class] = []
        result_dict_all[test_class] = {}
        print("Testing on class", test_class)
        result_dict_all[test_class]["angle"] = []
        result_dict_all[test_class]["transl"] = []
        result_dict_all[test_class]["ADDL2"] = []
        with torch.no_grad():
            for i, (init_imgs, gt_imgs, T_CO_init, T_CO_gt, mesh_verts, mesh_paths, depths, cam_mats) in enumerate(data_loader):
                #print(T_CO_gt.shape)
                print("len", np.linalg.norm(T_CO_gt[0,:3,3]))
                gt_imgs_raster,_ = render_batch(T_CO_gt.numpy(), mesh_paths, cam_mats.numpy(), img_size, use_par_render)
                ex_logdir = os.path.join(test_class_logdir, "ex"+format(i, "03d"))
                os.makedirs(ex_logdir, exist_ok=True)
                depths = depths.numpy()
                bsz = len(T_CO_init)
                T_CO_pred = T_CO_init
                gt_imgs = gt_imgs.numpy()
                mesh_verts = mesh_verts.to(device)
                T_CO_gt = T_CO_gt.to(device)
                pred_imgs_seq = []
                result_list = []

                
                for j in range(test_predict_iterations):
                    start_time = time.time()
                    print(test_class, "ex", i, "predict iter", j)
                    if(j==0):
                        pred_imgs = init_imgs.numpy()
                        T_CO_pred = T_CO_pred.to(device)
                    else:
                        pred_imgs, depths = render_batch(T_CO_pred, mesh_paths, cam_mats.numpy(), img_size, use_par_render)
                        T_CO_pred = torch.tensor(T_CO_pred).to(device)
                        pred_imgs_seq.append(pred_imgs)
                    model_input = prepare_model_input(pred_imgs, gt_imgs, depths, use_norm_depth).to(device)
                    model_output = model(model_input)
                    T_CO_pred_new = calculate_T_CO_pred(model_output, T_CO_pred, rotation_repr, cam_mats)
                    end_time = time.time()
                    addl1_loss = compute_ADD_L1_loss(T_CO_gt, T_CO_pred_new, mesh_verts, use_batch_mean=False)
                    addl1_loss = addl1_loss.detach().cpu().numpy()
                    addl2_loss = compute_ADD_L2_loss(T_CO_gt, T_CO_pred_new, mesh_verts, use_batch_mean=False)
                    addl2_loss = addl2_loss.detach().cpu().numpy()
                    angular_dist = compute_angular_dist(T_CO_gt, T_CO_pred)*180/np.pi
                    transl_dist = compute_transl_dist(T_CO_gt, T_CO_pred)
                    print("Add l2 loss", addl2_loss)
                    result_list.append(float(addl2_loss[0]))

                    #print("Add l1 loss", addl1_loss)
                    T_CO_pred = T_CO_pred_new.detach().cpu().numpy()
                    results[j][examples:(examples+bsz)] = addl2_loss
                    duration = end_time-start_time
                    iteration_timings.append(duration)
                    if j==(test_predict_iterations-1):
                        result_dict_all[test_class]["angle"].append(angular_dist)
                        result_dict_all[test_class]["transl"].append(float(transl_dist))
                        result_dict_all[test_class]["ADDL2"].append(float(addl2_loss))


                    

                pred_imgs, depths = render_batch(T_CO_pred, mesh_paths, cam_mats.numpy(), img_size, use_par_render)
                pred_imgs_seq.append(pred_imgs)
                examples += bsz
                result_dict[test_class].append(result_list)
                print("len pred imgs seq")
                print(len(pred_imgs_seq))
                create_interpolated_image_fig(init_imgs.numpy(), gt_imgs, pred_imgs_seq, gt_imgs_raster, save_dir=ex_logdir, train_val_or_test="imgs")
        mean_add_loss = np.mean(results, axis=1)

        

        

        median_add_loss = np.median(results, axis=1)
        last_iter_median_add_loss = median_add_loss[-1]
        print("last iter median add loss")
        print(last_iter_median_add_loss)
        last_iter_add_losses = results[-1,:]
        print("laster iter add losses")
        print(last_iter_add_losses)
        print("mean_add_loss", mean_add_loss)
        print("last iter meian add loss")
        print(last_iter_median_add_loss)
        print("example closest to mean")
        idx_close_mean = find_nearest(last_iter_add_losses, last_iter_median_add_loss)
        print(idx_close_mean)
        mean_examples[test_class] = {}
        mean_examples[test_class]["median_idx"] = idx_close_mean
        mean_examples[test_class]["val"] = last_iter_add_losses[idx_close_mean]
        mean_examples[test_class]["median_val"] = last_iter_median_add_loss
        plt.bar(np.arange(len(mean_add_loss)), mean_add_loss)
        plt.savefig(os.path.join(test_class_logdir, "bar_plot.png"))
        plt.close()
        plt.bar(np.arange(len(median_add_loss)), median_add_loss)
        plt.savefig(os.path.join(test_class_logdir, "median_bar_plot.png"))
        plt.close()
    

    for test_class in result_dict_all:
        angles = np.array(result_dict_all[test_class]["angle"])
        transls = np.array(result_dict_all[test_class]["transl"])
        addl2 = np.array(result_dict_all[test_class]["ADDL2"])
        idx_sorted_addl2 = np.argsort(result_dict_all[test_class]["ADDL2"])
        print("mean angle dev", test_class, np.mean(angles))
        result_dict_all[test_class]["angle_mean"] = float(np.mean(angles))
        result_dict_all[test_class]["transl_mean"] = float(np.mean(transls))
        result_dict_all[test_class]["ADDL2_mean"] = float(np.mean(addl2))
        print(test_class)
        print("idx sorted addl2")
        #print(idx_sorted_addl2)
        idx_sorted_ex = [f'ex{format(idx, "04d")}' for idx in idx_sorted_addl2]
        print(idx_sorted_ex)
    #print("result_dict_all")
    #print(result_dict_all)
    print("idx sorted addl2")
    print(idx_sorted_addl2)


    print("mean examples")
    print(mean_examples)
    print("avg time")
    print(np.array(iteration_timings).mean())
    write_json(os.path.join(test_logdir, "results.json"), result_dict)
    write_json(os.path.join(test_logdir, "results_all.json"), result_dict_all)
    write_pickle(os.path.join(test_logdir, "results.pkl"), result_dict)
    return np.mean(results,axis=1)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


if __name__ == '__main__':
    print("Testig model")
    config = get_dict_from_cli()
    # model load parameters
    model_name = config["network"]["backend_network"]
    rotation_repr = config["network"]["rotation_representation"]
    use_pretrained = config["model_io"]["use_pretrained_model"]
    use_pretrained = True
    model_save_dir = config["model_io"]["model_save_dir"]
    model_save_name = config["model_io"]["model_save_name"]
    pretrained_path = os.path.join(model_save_dir, model_save_name)
    print("Pretrained path", pretrained_path)
    use_norm_depth = config["advanced"]["use_normalized_depth"]
    model = fetch_network(model_name, rotation_repr, use_norm_depth, use_pretrained, pretrained_path)
    logdir = config["logging"]["logdir"]
    os.makedirs(logdir, exist_ok=True)
    # print training info
    print("")
    print(" ### TESTING IS STARTING ### ")
    print("Loading backend network", model_name.upper(), "with rotation representation", rotation_repr)
    mean_loss = test_model(model, config)
    print("Mean loss", mean_loss)
    #save_loss_bar_plot(loss_dict, logdir)



