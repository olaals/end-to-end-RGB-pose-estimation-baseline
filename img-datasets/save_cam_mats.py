import numpy as np
import os

def get_camera_matrix(intrinsics):
    focal_len = intrinsics["focal_length"]
    img_res = intrinsics["image_resolution"]
    sensor_width = intrinsics["sensor_width"]
    pix_per_mm = sensor_width/img_res
    fx = fy = focal_len/pix_per_mm
    vx = vy = img_res/2
    K = np.array([[fx, 0, vx],[0, fy, vy],[0,0,1]])
    return K

cam_intr={
    "focal_length": 50, #mm
    "sensor_width": 36, #mm
    "image_resolution": 320, # width=height
}

ds_name = "MN10-tless-30k"
K = get_camera_matrix(cam_intr)


class_dirs = [os.path.join(ds_name, classname) for classname in os.listdir(ds_name)]
for class_dir in class_dirs:
    if(not os.path.isdir(class_dir)):
        continue
    dataset_class_types = [os.path.join(class_dir, class_type) for class_type in os.listdir(class_dir)]
    for dataset_class_type in dataset_class_types:
        print(dataset_class_types)
        example_paths = [os.path.join(dataset_class_type, ex_dir_name) for ex_dir_name in os.listdir(dataset_class_type)]
        for example_path in example_paths:
            np.save(os.path.join(example_path, "K.npy"), K)
            #print(np.load(os.path.join(example_path, "K.npy")))














