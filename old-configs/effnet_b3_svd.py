import torch
import os

all_classes_modelnet10 = ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table","toilet"]


def get_config():

    this_file_name = os.path.split(os.path.splitext(__file__)[0])[-1]
    print("Config file name:", this_file_name)

    rotation_rep = "SVD" #SVD or 6D,
    backend_network = "effnet_b3"
    img_dataset = "MN10-texture-30k"
    model3d_dataset = "ModelNet10-texturized"




    return {
        "config_name":this_file_name,
        "train_params":{
            "batch_size":8,
            "learning_rate": 3e-4, 
            "num_batches_to_train": 200000, # stop training after N batches
            "optimizer":"adam",
            "loss": "add_l1_disentangled",
            "num_sample_vertices": 1000,  # number of vertices sampled from the mesh, used in calculating the loss
            "device": "cuda", # cuda or cpu 
        },
        "dataset_config":{
            "train_from_images": True,
            "train_classes": all_classes_modelnet10, # all_classes or specify indivudal as ["desk", "sofa", "plant"]
            "model3d_dataset": model3d_dataset,
            "img_dataset": img_dataset,
            "img_ds_conf":{
                "real": "real.png",
                "init": "init.png"
            },
        },
        "network":{
            "backend_network": backend_network,
            "rotation_representation": rotation_rep, #SVD or 6D, 
        },
        "camera_intrinsics":{
            "focal_length": 50, #mm
            "sensor_width": 36, #mm
            "image_resolution": 320, # width=height
        },
        "scene_config":{
            "distance_cam_to_world": 2.5, #meters
            "distance_cam_to_world_deviation":0.1, #meters
            "world_to_object_gt_transl_deviation": 0.1, #meters
            "world_to_object_transl_deviation": 0.1, #meters
            "world_to_object_angle_deviation":30, #degrees
            "use_parallel_rendering":False,
        },
        "model_io":{
            "use_pretrained_model": False,  # start training from a pretrained model
            "pretrained_model_name": "", # load predtrained model, if use_pretrained_model = True
            "model_save_dir": os.path.join("models", "saved-models", img_dataset),
            "model_save_name": this_file_name + "-" + backend_network+"-"+rotation_rep+".pth",
            "batch_model_save_interval": 500,  # save model during tranining after every N batch trained
        },
        "logging":{
            "logdir": os.path.join("logdir", img_dataset, this_file_name),
            "save_viz_every_n_batch": 5000,
            "save_visualization_at_batches": [100, 500, 1000, 2000],
            "log_save_interval":50,
            "validation_interval":1000,
            "val_examples_from_each_class":8,
        },
        "test_config":{
            "batch_size": 8, 
            "predict_iterations": 3,
            "iterations_per_class": 1,
            "model_load_dir": os.path.join("models", "saved-models"),
            "model_load_name": this_file_name + "-" + backend_network+"-"+rotation_rep+".pth",
            "test_classes": all_classes_modelnet10,
        },
        "advanced":{
            "use_normalized_depth": False, # use a normalized rendered depth in the model input
            "train_iter_policy": "incremental", # constant or incremental
            "train_iter_policy_argument": [(150000,2),(180000,3)], # if train_iter_policy is constant use a number i.e. 3, if incremental use tuple list [(100,2),(1000,3)]
        },


    }


if __name__ == '__main__':
    config = get_config()
    for param_dict_key in config:
        param_dict = config[param_dict_key]
        print("")
        print(param_dict_key.upper())
        for key in param_dict:
            value = param_dict[key]
            print(key, ":", value)
