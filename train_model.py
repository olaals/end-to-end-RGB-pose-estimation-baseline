from config import get_config
from models.baseline_net import BaseNet
import torch
from data_loaders import get_train_batch
from loss import compute_ADD_L1_loss
from rotation_representation import calculate_T_CO_pred
#from models.efficient_net import 
from models import fetch_network
import os






def init_training():
    config = get_config()
    scene_config = config["scene_config"]

    ds_name = config["train_params"]["dataset_name"]
    train_classes = config["train_params"]["train_classes"]

    # model load parameters
    model_name = config["network"]["backend_network"]
    rotation_repr = config["network"]["rotation_representation"]
    device = config["train_params"]["device"]
    use_pretrained = config["model_io"]["use_pretrained_model"]
    # model saving
    save_every_n_batch = config["model_io"]["batch_model_save_interval"]
    model_save_dir = config["model_io"]["model_save_dir"]
    model_save_name = config["model_io"]["model_save_name"]
    model_save_path = os.path.join(model_save_dir, model_save_name)

    cam_intrinsics = config["camera_intrinsics"]
    print("Loading backend network", model_name.upper(), "with rotation representation", rotation_repr)
    
    model = fetch_network(model_name, rotation_repr)
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



    """
    TRAINING LOOP
    """
    for i in range(num_train_batches):
        optimizer.zero_grad()
        model_input, T_CO_init, T_CO_gt, mesh_verts, cam_mats = get_train_batch(
                ds_name, batch_size, train_classes, num_sample_verts, cam_intrinsics, scene_config, device)

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
    init_training()




