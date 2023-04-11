import torch
import scipy
from scipy import ndimage, misc
import numpy as np
import trimesh as tm
import random
import os
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
from se3_helpers import get_T_CO_init_and_gt
from renderer import render_scene
from torch.utils.data import Dataset, DataLoader
import threading
import time
from PIL import Image
import yaml
transforms = A.Compose([
    ToTensorV2()
])

def get_dataset_class_paths(dataset_path, classname, train_val_or_test):
    data_dir = os.path.join(dataset_path, classname, train_val_or_test)
    file_paths = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
    return file_paths

def get_ds_file_paths(ds_path, classes, train_val_or_test):
    all_training_paths = []
    for classname in classes:
        class_train_paths = get_dataset_class_paths(ds_path, classname, train_val_or_test)
        all_training_paths = all_training_paths + class_train_paths
    return all_training_paths

def get_mesh_path_from_yaml(yaml_path):
    with open(yaml_path) as file:
        metadata = yaml.load(file, Loader=yaml.FullLoader)
        ds_name = metadata["dataset_name"]
        mesh_class = metadata["mesh_class"]
        mesh_filename = metadata["mesh_filename"]
        train_or_test = metadata["train_or_test"]
        mesh_path = os.path.join("model3d-datasets", ds_name, mesh_class, train_or_test, mesh_filename)
    return mesh_path

def get_gradient(img):
    # img to gray
    img = np.array(img)
    img = np.mean(img, axis=2)
    result = ndimage.gaussian_gradient_magnitude(img, sigma=1)
    result = np.dstack((result, result, result))
    # divide by max
    result = result / np.max(result)
    return result


class ImagePoseDataset(Dataset):
    def __init__(self, train_val_or_test, ds_conf):
        dataset_path = os.path.join("img-datasets", ds_conf["img_dataset"])
        classes = ds_conf["classes"]
        self.train_val_or_test = train_val_or_test
        self.all_paths = get_ds_file_paths(dataset_path, classes, train_val_or_test )
        self.all_paths.sort()
        self.ds_conf = ds_conf
        self.filename_real = ds_conf["img_ds_conf"]["real"]
        self.filename_init = ds_conf["img_ds_conf"]["init"]

    def __len__(self):
        return len(self.all_paths)


    def __getitem__(self, idx):
        real_path = os.path.join(self.all_paths[idx], self.filename_real)
        init_path = os.path.join(self.all_paths[idx], self.filename_init)
        verts = np.load(os.path.join(self.all_paths[idx], "vertices.npy"))
        T_CO_gt = np.load(os.path.join(self.all_paths[idx], "T_CO_gt.npy"))
        T_CO_gt = T_CO_gt.astype(np.float32)
        T_CO_init = np.load(os.path.join(self.all_paths[idx], "T_CO_init.npy"))
        depth_pass = np.load(os.path.join(self.all_paths[idx], "init_depth.npy"))
        K = np.load(os.path.join(self.all_paths[idx], "K.npy"))
        real_img = np.asarray(Image.open(real_path).convert('RGB'))
        #real_img = get_gradient(real_img)
        #if "augment" in self.ds_conf:
            #real_img = self.ds_conf["augment"](image=real_img)["image"]
        real_img = real_img/255.0
        init_img = np.asarray(Image.open(init_path))
        #init_img = A.RandomBrightnessContrast()(image=init_img)["image"]

        if np.max(init_img) > 0.0:
            init_img = init_img/np.max(init_img)
        mesh_path = get_mesh_path_from_yaml(os.path.join(self.all_paths[idx], "metadata.yml"))
        return init_img, real_img, T_CO_init, T_CO_gt, verts, mesh_path, depth_pass, K

def get_dataloaders(ds_conf, batch_size):
    train_ds = ImagePoseDataset("train", ds_conf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_ds = ImagePoseDataset("validation", ds_conf)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_ds = ImagePoseDataset("test", ds_conf)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def get_dataloader(ds_conf, batch_size, train_test_or_val="train"):
    if(train_test_or_val == 'train'):
        train_ds = ImagePoseDataset("train", ds_conf)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        return train_loader
    elif(train_test_or_val == 'val'):
        val_ds = ImagePoseDataset("validation", ds_conf)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
        return val_loader
    elif(train_test_or_val == 'test'):
        test_ds = ImagePoseDataset("test", ds_conf)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        return test_loader
    else:
        assert False, f'Could not find dataloader for {train_test_or_val}'





if __name__ == '__main__':
    import sys
    sys.path.append("configs")
    from baseline_cfg import get_config
    config = get_config()
    ds_conf = config["dataset_config"]
    dataset_path = "img-datasets/MN10-alu-1k"
    classes = ["chair", "bed"]

    train, val, test = get_dataloaders(ds_conf, 8)
    while(True):
        a = next(iter(train))
        for i in a:
            print(type(i))
            try: 
                print(i.shape)
            except:
                print(len(i))


