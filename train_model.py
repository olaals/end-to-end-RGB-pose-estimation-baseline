from config import get_config
#from models.baseline_net import BaseNet
import torch
from data_loaders import *
from loss import compute_ADD_L1_loss
from rotation_representation import calculate_T_CO_pred
#from models.efficient_net import 
from models import fetch_network
import os



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
    for i in range(num_train_batches):
        optimizer.zero_grad()

        T_CO_init, T_CO_gt = sample_T_CO_inits_and_gts(batch_size, scene_config)
        mesh_paths = sample_mesh_paths(batch_size, ds_name, train_classes, "train")
        init_imgs, norm_depth = render_batch(T_CO_init, mesh_paths, cam_intrinsics)
        if not use_norm_depth: norm_depth = None
        gt_imgs,_ = render_batch(T_CO_gt, mesh_paths, cam_intrinsics)
        model_input = prepare_model_input(init_imgs, gt_imgs, norm_depth).to(device)
        cam_mats = get_camera_mat_tensor(cam_intrinsics, batch_size).to(device)
        mesh_verts = sample_verts_to_batch(mesh_paths, num_sample_verts).to(device) 
        
        T_CO_init = torch.tensor(T_CO_init).to(device)
        T_CO_gt = torch.tensor(T_CO_gt).to(device)
        model_output = model(model_input)
        T_CO_pred = calculate_T_CO_pred(model_output, T_CO_init, rotation_repr, cam_mats)
        loss = compute_ADD_L1_loss(T_CO_gt, T_CO_pred, mesh_verts)
        loss.backward()
        optimizer.step()
        print(f'Loss for train batch {i}:', loss.item())
        if i != 0 and i%save_every_n_batch == 0:
            print("Saving model to", model_save_path)
            torch.save(model.state_dict(), model_save_path)
    """
    END TRAIN LOOP
    """




if __name__ == '__main__':
    config = get_config()
    train(config)




