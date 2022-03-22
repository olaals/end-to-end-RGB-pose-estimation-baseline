import torch
import os


all_classes_modelnet40 = ["airplane", "bench", "bowl", "cone", "desk", "flower_pot", "keyboard", "mantel", "person", "radio",
                          "sofa", "table", "tv_stand", "xbox", "bathtub", "bookshelf", "car", "cup", "door", "glass_box",
                          "lamp", "monitor", "piano", "range_hood", "stairs", "tent", "vase", "bed", "bottle", "chair", "curtain",
                          "dresser", "guitar", "laptop", "night_stand", "plant", "sink", "stool", "toilet", "wardrobe"]


def get_config():
    return {
        "train_params":{
            "batch_size":8,
            "train_classes": all_classes_modelnet40, # all_classes or specify indivudal as ["desk", "sofa", "plant"]
            "learning_rate": 3e-4,
            "optimizer":"adam",
            "num_sample_vertices": 1000,
        },
        "network_details":{
            "backend_network": "baseline",
            "rotation_representation": "SVD", #SVD or 6D, 
        },
        "camera_intrinsics":{
            "focal_length": 50, #mm
            "sensor_width": 36, #mm
            "image_resolution": 300, # width=height
        },
        "scene_positioning":{
            "distance_cam_to_world": 1.5,
            "distance_cam_to_world_deviation":0.1,
            "world_to_object_gt_transl_deviation": 0.1,
            "world_to_object_transl_deviation": 0.1,
            "world_to_object_angle_deviation":25,
        },
        "model_io":{
            "use_pretrained_model": False,
            "pretrained_model_path": os.path.join("saved_models", "sdffds"),
            "model_save_path": os.path.join("saved_models", "baseline_state_dict.pth",
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
            print(key, value)
