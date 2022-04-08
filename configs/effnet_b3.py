import torch
import os


all_classes_modelnet40 = ["airplane", "bench", "bowl", "cone", "desk", "flower_pot", "keyboard", "mantel", "person", "radio",
                          "sofa", "table", "tv_stand", "xbox", "bathtub", "bookshelf", "car", "cup", "door", "glass_box",
                          "lamp", "monitor", "piano", "range_hood", "stairs", "tent", "vase", "bed", "bottle", "chair", "curtain",
                          "dresser", "guitar", "laptop", "night_stand", "plant", "sink", "stool", "toilet", "wardrobe"]


def get_config():

    this_file_name = os.path.split(os.path.splitext(__file__)[0])[-1]
    print("Config file name:", this_file_name)

    rotation_rep = "SVD" #SVD or 6D,
    backend_network = "effnet_b3"




    return {
        "config_name":this_file_name,
        "train_params":{
            "batch_size":8,
            "train_classes": all_classes_modelnet40, # all_classes or specify indivudal as ["desk", "sofa", "plant"]
            "learning_rate": 3e-4, 
            "num_batches_to_train": 50000, # stop training after N batches
            "optimizer":"adam",
            "loss": "add_l1_disentangled",
            "num_sample_vertices": 1000,  # number of vertices sampled from the mesh, used in calculating the loss
            "device": "cuda", # cuda or cpu 
            "dataset_name": "ModelNet40-norm-ply",
        },
        "network":{
            "backend_network": backend_network,
            "rotation_representation": rotation_rep, #SVD or 6D, 
        },
        "camera_intrinsics":{
            "focal_length": 50, #mm
            "sensor_width": 36, #mm
            "image_resolution": 300, # width=height
        },
        "scene_config":{
            "distance_cam_to_world": 1.8, #meters
            "distance_cam_to_world_deviation":0.1, #meters
            "world_to_object_gt_transl_deviation": 0.1, #meters
            "world_to_object_transl_deviation": 0.1, #meters
            "world_to_object_angle_deviation":25, #degrees
        },
        "model_io":{
            "use_pretrained_model": False,  # start training from a pretrained model
            "pretrained_model_name": "", # load predtrained model, if use_pretrained_model = True
            "model_save_dir": os.path.join("models", "saved-models"),
            "model_save_name": this_file_name + "-" + backend_network+"-"+rotation_rep+".pth",
            "batch_model_save_interval": 25,  # save model during tranining after every N batch trained
        },
        "logging":{
            "logdir": os.path.join("logdir", this_file_name),
            "save_visualization_at_batches": [100, 500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000, 70000, 90000],
            "log_save_interval":10,
            "validation_interval":250,
            "val_examples_from_each_class":8,

        },
        "test_config":{
            "batch_size": 8, 
            "predict_iterations": 3,
            "iterations_per_class": 1,
            "model_load_dir": os.path.join("models", "saved-models"),
            "model_load_name": this_file_name + "-" + backend_network+"-"+rotation_rep+".pth",
            "test_classes": all_classes_modelnet40,
        },
        "advanced":{
            "use_normalized_depth": False, # use a normalized rendered depth in the model input
            "train_iter_policy": "incremental", # constant or incremental
            "train_iter_policy_argument": [(10000,2), (30000,3)], # if train_iter_policy is constant use a number i.e. 3, if incremental use tuple list [(100,2),(1000,3)]
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
