#from models.baseline_net import BaseNet
import torch
from data_loaders import *
from loss import compute_ADD_L1_loss, compute_disentangled_ADD_L1_loss
from rotation_representation import calculate_T_CO_pred
#from models.efficient_net import 
from models import fetch_network
import os
from config_parser import get_dict_from_cli




def train(config):
    scene_config = config["scene_config"]

    ds_name = config["train_params"]["dataset_name"]
    train_classes = config["train_params"]["train_classes"]

    # model load parameters
    model_name = config["network"]["backend_network"]
    rotation_repr = config["network"]["rotation_representation"]
    device = config["train_params"]["device"]
    use_pretrained = config["model_io"]["use_pretrained_model"]
    model_save_dir = config["model_io"]["model_save_dir"]
    pretrained_name = config["model_io"]["pretrained_model_name"]
    pretrained_path = os.path.join(model_save_dir, pretrained_name)
    use_norm_depth = config["advanced"]["use_normalized_depth"]
    # model saving
    save_every_n_batch = config["model_io"]["batch_model_save_interval"]
    model_save_name = config["model_io"]["model_save_name"]
    model_save_path = os.path.join(model_save_dir, model_save_name)

    cam_intrinsics = config["camera_intrinsics"]
    
    model = fetch_network(model_name, rotation_repr, use_norm_depth, use_pretrained, pretrained_path)
    model = model.to(device)

    #train params
    batch_size = config["train_params"]["batch_size"]
    learning_rate = config["train_params"]["learning_rate"]
    opt_name = config["train_params"]["optimizer"]
    num_train_batches = config["train_params"]["num_batches_to_train"]
    num_sample_verts = config["train_params"]["num_sample_vertices"]
    device = config["train_params"]["device"]
    use_disentangled_loss = config["advanced"]["use_disentangled_loss"]

    # train iteration policy, i.e. determine how many iterations per batch
    train_iter_policy_name = config["advanced"]["train_iter_policy"]
    policy_argument = config["advanced"]["train_iter_policy_argument"]
    if train_iter_policy_name == 'constant':
        train_iter_policy = train_iter_policy_constant
    elif train_iter_policy_name == 'incremental':
        train_iter_policy = train_iter_policy_incremental
    

    if(opt_name == "adam"):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif(opt_name == 'sgd'):
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        assert False

    # print training info
    print("")
    print(" ### TRAINING IS STARTING ### ")
    print("Loading backend network", model_name.upper(), "with rotation representation", rotation_repr)
    print("Batch size", batch_size, "Learning rate", learning_rate, "Optimizer", opt_name.upper())
    print("Training on device", device)
    if use_pretrained:
        print("Pretrained model is loaded from", predtrained_path)
    else:
        print("No pretrained model used, training from scratch")
    print("The model will be saved to", model_save_path)
    if use_norm_depth:
        print("The model is trained with the normalized depth from the CAD model (advanced)")
    print("")



    """
    TRAINING LOOP
    """
    train_examples=0
    batch_num = 0
    while(True):
        T_CO_init, T_CO_gt = sample_T_CO_inits_and_gts(batch_size, scene_config)
        mesh_paths = sample_mesh_paths(batch_size, ds_name, train_classes, "train")
        mesh_verts = sample_verts_to_batch(mesh_paths, num_sample_verts).to(device) 
        cam_mats = get_camera_mat_tensor(cam_intrinsics, batch_size).to(device)
        gt_imgs,_ = render_batch(T_CO_gt, mesh_paths, cam_intrinsics)
        T_CO_gt = torch.tensor(T_CO_gt).to(device)

        T_CO_pred = T_CO_init # current pred is initial
        train_iterations = train_iter_policy(batch_num, policy_argument)
        for j in range(train_iterations):
            optimizer.zero_grad()
            pred_imgs, norm_depth = render_batch(T_CO_pred, mesh_paths, cam_intrinsics)
            model_input = prepare_model_input(pred_imgs, gt_imgs, norm_depth, use_norm_depth).to(device)
            T_CO_pred = torch.tensor(T_CO_pred).to(device)
            model_output = model(model_input)
            T_CO_pred_new = calculate_T_CO_pred(model_output, T_CO_pred, rotation_repr, cam_mats)
            if not use_disentangled_loss:
                loss = compute_ADD_L1_loss(T_CO_gt, T_CO_pred_new, mesh_verts)
            else:
                loss = compute_disentangled_ADD_L1_loss(T_CO_gt, T_CO_pred_new, mesh_verts)
            loss.backward()
            optimizer.step()

            T_CO_pred = T_CO_pred_new.detach().cpu().numpy()
            print(f'Loss for train batch {batch_num}, train iter {j}:', loss.item())
            train_examples=train_examples+1


        if batch_num != 0 and batch_num%save_every_n_batch == 0:
            print("Saving model to", model_save_path)
            torch.save(model.state_dict(), model_save_path)
            print("Model output")
            print(model_output)
        batch_num += 1
        if train_examples > num_train_batches:
            break
    """
    END TRAIN LOOP
    """


def train_iter_policy_constant(current_batch, num):
    return num

def train_iter_policy_incremental(current_batch, increments_tuple_list):
    # input must have form [(300, 2), (1000,3), (3000,4)] 
    new_train_iters = 1
    for (batch_num, train_iters) in increments_tuple_list:
        if (current_batch>batch_num):
            new_train_iters = train_iters
    return new_train_iters





if __name__ == '__main__':
    try:
        config = get_dict_from_cli()
    except:
        raise Exception("Include a valid config file with: ".upper()+"python train_model.py baseline_cfg")
    train(config)





