#from models.baseline_net import BaseNet
import torch 
from data_loaders import *
from image_dataloaders import get_dataloaders
from loss import compute_ADD_L1_loss, compute_disentangled_ADD_L1_loss, compute_scaled_disentl_ADD_L1_loss
from rotation_representation import calculate_T_CO_pred
#from models.efficient_net import 
from models import fetch_network
import os
from parser_config import get_dict_from_cli
import pickle
import matplotlib.pyplot as plt
from visualization import visualize_examples
from test_model import evaluate_model, validate_model
from torch.utils.tensorboard import SummaryWriter
import time
import datetime

def pickle_log_dict(log_dict, logdir):
    save_path = os.path.join(logdir, "log_dict.pkl")
    with open(save_path, 'wb') as handle:
        pickle.dump(log_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def calculate_eta(start_time, perc_complete):
    curr_time = time.time()
    sec_since_start = curr_time - start_time
    est_total_time = sec_since_start/perc_complete
    est_remaining = est_total_time-sec_since_start
    return str(datetime.timedelta(seconds=est_remaining))
    

def save_loss_plot(losses, training_examples, loss_name, logdir):
    assert len(losses) == len(training_examples)
    fig,ax = plt.subplots()
    fig.set_size_inches(9.5, 5.5)
    ax.set_title(loss_name)
    ax.set_xlabel("Training examples")
    ax.set_ylabel(loss_name)
    ax.set_yscale('log')
    plt.plot(training_examples, losses)
    save_path = os.path.join(logdir, loss_name.replace(" ", "-")+".png")
    plt.savefig(save_path)
    plt.close()

def save_plot_validation_loss(val_data_struct,logdir, loss_name):
    fig,ax = plt.subplots()
    fig.set_size_inches(9.5, 5.5)
    ax.set_title("Validation " + loss_name)
    ax.set_xlabel("Training examples")
    ax.set_ylabel(loss_name)
    ax.set_yscale('log')
    train_exs = []
    val_losses_arr = []
    for (train_ex, val_losses) in val_data_struct:
        train_exs.append(train_ex)
        val_losses_arr.append(val_losses)
    val_losses = np.array(val_losses_arr)
    train_exs = np.array(train_exs)
    legends = []
    for pred_iter in range(val_losses.shape[1]):
        legends.append("Pred.iter"+str(pred_iter+1))
        iter_val_losses = val_losses[:,pred_iter]
        plt.plot(train_exs,iter_val_losses, label="Iter. "+str(pred_iter))
    ax.legend(legends)
    save_path = os.path.join(logdir, "validation-"+loss_name.replace(" ", "-")+".png")
    plt.savefig(save_path)
    plt.close()
        


def logging(model, config, writer, log_dict, logdir, batch_num, train_examples):
    log_interval = config["logging"]["log_save_interval"]
    if(batch_num%log_interval == 0):
        current_loss = log_dict["loss"]["add_l1"][:batch_num]
        current_train_ex =log_dict["loss"]["train_ex"][:batch_num] 
        save_loss_plot(current_loss, current_train_ex, "ADD L1 Loss", logdir)
        pickle_log_dict(log_dict, logdir)
    
    save_viz_batches = config["logging"]["save_visualization_at_batches"]
    save_viz_every_n_batch = config["logging"]["save_viz_every_n_batch"]
    if((batch_num in save_viz_batches) or (batch_num%save_viz_every_n_batch==0 and batch_num!=0)):
        viz_dir = os.path.join(logdir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        viz_save_path  = os.path.join(viz_dir, "viz-at-train-ex-"+str(train_examples)+".png")
        visualize_examples(model, config, "train", show_fig=False, save_fig=True, save_path=viz_save_path)

    validation_interval = config["logging"]["validation_interval"]
    if(batch_num%validation_interval == 0 and batch_num != 0):
        val_ex = config["logging"]["val_examples_from_each_class"]
        #loss_dict, mean_losses = evaluate_model(model, config, "train", use_all_examples=False, max_examples_from_each_class=val_ex)
        mean_losses = validate_model(model, config, "val")
        #log_dict["val_loss_dicts"].append((train_examples, loss_dict))
        log_dict["val_loss"].append((train_examples, mean_losses))
        pickle_log_dict(log_dict, logdir)
        save_plot_validation_loss(log_dict["val_loss"], logdir, "ADD L1 loss")

        #tensorboard
        iter_dict = {}
        for i in range(len(mean_losses)):
            writer.add_scalar(f'Validation_ADD_L1_loss/Iter{i}', mean_losses[i], train_examples)
            iter_dict[f'Iter{i}'] = mean_losses[i]
        writer.add_scalars('Validation_ADD_L1_loss_iters', iter_dict, train_examples)


    model.train()



def train(config):
    scene_config = config["scene_config"]

    # dataset config
    model3d_dataset = config["dataset_config"]["model3d_dataset"]
    train_classes = config["dataset_config"]["train_classes"]
    train_from_imgs = config["dataset_config"]["train_from_images"]
    ds_conf = config["dataset_config"]
    batch_size = config["train_params"]["batch_size"]
    if train_from_imgs:
        train_loader, val_loader, test_loader = get_dataloaders(ds_conf, batch_size)


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
    learning_rate = config["train_params"]["learning_rate"]
    opt_name = config["train_params"]["optimizer"]
    num_train_batches = config["train_params"]["num_batches_to_train"]
    num_sample_verts = config["train_params"]["num_sample_vertices"]
    device = config["train_params"]["device"]
    loss_fn_name = config["train_params"]["loss"]

    # train iteration policy, i.e. determine how many iterations per batch
    train_iter_policy_name = config["advanced"]["train_iter_policy"]
    policy_argument = config["advanced"]["train_iter_policy_argument"]
    if train_iter_policy_name == 'constant':
        train_iter_policy = train_iter_policy_constant
    elif train_iter_policy_name == 'incremental':
        train_iter_policy = train_iter_policy_incremental
    else:
        assert False

    # parallel rendering
    use_par_render = config["scene_config"]["use_parallel_rendering"]
    

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
        print("Pretrained model is loaded from", pretrained_path)
    else:
        print("No pretrained model used, training from scratch")
    print("The model will be saved to", model_save_path)
    if use_norm_depth:
        print("The model is trained with the normalized depth from the CAD model (advanced)")
    print("")
    
    # logging
    log_dict = {}
    log_dict["loss"] = {}
    log_dict["loss"]["add_l1"] = np.zeros((num_train_batches+1))
    log_dict["loss"]["train_ex"] = np.zeros((num_train_batches+1))
    log_dict["val_loss_dicts"] = []
    log_dict["val_loss"] = []
    logdir = config["logging"]["logdir"]
    os.makedirs(logdir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=os.path.join("tensorboard", config["config_name"]))
    
    start_time = time.time()


    """
    TRAINING LOOP
    """
    train_examples=0
    new_batch_num=0
    batch_num=0

    while(True):
        start_time = time.time()
        if train_from_imgs:
            init_imgs, gt_imgs, T_CO_init, T_CO_gt, mesh_verts, mesh_paths = next(iter(train_loader))
            init_imgs = init_imgs.numpy()
            gt_imgs = gt_imgs.numpy()
            T_CO_gt = T_CO_gt.to(device)
            mesh_verts = mesh_verts.to(device)
        else:
            T_CO_init, T_CO_gt = sample_T_CO_inits_and_gts(batch_size, scene_config)
            mesh_paths = sample_mesh_paths(batch_size, model3d_dataset, train_classes, "train")
            mesh_verts = sample_verts_to_batch(mesh_paths, num_sample_verts).to(device) 
            gt_imgs,_ = render_batch(T_CO_gt, mesh_paths, cam_intrinsics, use_par_render)
            T_CO_gt = torch.tensor(T_CO_gt).to(device)

        cam_mats = get_camera_mat_tensor(cam_intrinsics, batch_size).to(device)
        T_CO_pred = T_CO_init # current pred is initial
        train_iterations = train_iter_policy(batch_num, policy_argument)
        for j in range(train_iterations):
            optimizer.zero_grad()
            if(j==0 and train_from_imgs):
                pred_imgs = init_imgs
                norm_depth = None
                T_CO_pred = T_CO_pred.to(device)
            else:
                pred_imgs, norm_depth = render_batch(T_CO_pred, mesh_paths, cam_intrinsics, use_par_render)
                T_CO_pred = torch.tensor(T_CO_pred).to(device)
            model_input = prepare_model_input(pred_imgs, gt_imgs, norm_depth, use_norm_depth).to(device)
            model_output = model(model_input)
            T_CO_pred_new = calculate_T_CO_pred(model_output, T_CO_pred, rotation_repr, cam_mats)
            addl1_loss = compute_ADD_L1_loss(T_CO_gt, T_CO_pred_new, mesh_verts)
            if loss_fn_name == "add_l1":
                addl1_loss.backward()
            elif loss_fn_name == "add_l1_disentangled":
                disentl_loss = compute_disentangled_ADD_L1_loss(T_CO_gt, T_CO_pred_new, mesh_verts)
                disentl_loss.backward()
            elif loss_fn_name == "add_l1_disentl_scaled":
                sc_disentl_loss = compute_scaled_disentl_ADD_L1_loss(T_CO_pred, T_CO_pred_new, T_CO_gt, mesh_verts)
                sc_disentl_loss.backward()

            optimizer.step()
            T_CO_pred = T_CO_pred_new.detach().cpu().numpy()

            # Printing and logging
            elapsed = time.time() - start_time
            print(f'ADD L1 loss for train batch {batch_num}, with {new_batch_num} new batches, train iter {j}: {addl1_loss.item():.4f}, batch time: {elapsed:.3f}')
            log_dict["loss"]["add_l1"][batch_num] = addl1_loss.item()
            log_dict["loss"]["train_ex"][batch_num] = train_examples
            logging(model, config, writer, log_dict, logdir, batch_num, train_examples)
            writer.add_scalar("ADD_L1_loss", addl1_loss.item(), train_examples)

            if batch_num != 0 and batch_num%save_every_n_batch == 0:
                perc_complete = (batch_num*1.0)/num_train_batches
                print("Saving model to", model_save_path)
                print(f'Trained {batch_num} of {num_train_batches}. Training {(perc_complete*100.0):.3f} % complete.')
                print(f'Estimated remaining training time (hour,min,sec): {calculate_eta(start_time, perc_complete)}')
                torch.save(model.state_dict(), model_save_path)
            if batch_num >= num_train_batches:
                break
            train_examples=train_examples+batch_size
            batch_num += 1
        new_batch_num += 1
        if batch_num >= num_train_batches:
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
    config = get_dict_from_cli()
    train(config)

    
    






