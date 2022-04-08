import trimesh as tm
import numpy as np
import os


def center_mesh_to_centroid(mesh):
    c = mesh.centroid
    transf = np.eye(4)
    transf[:3, 3] = -c
    mesh.apply_transform(transf)

def rescale_mesh(mesh):
    bounding_box = mesh.extents
    longest_axis = np.max(bounding_box)
    transf_mat = np.eye(4)
    transf_mat[:3,:3] = transf_mat[:3, :3]/longest_axis
    mesh.apply_transform(transf_mat)

def decimate_mesh(mesh, ratio):
    current_faces = len(mesh.faces)
    new_faces = int(current_faces*ratio)
    simplified = mesh.simplify_quadratic_decimation(new_faces)
    return simplified



def change_extension(filename, new_extension):
    without_ext = os.path.splitext(filename)[0]
    with_new_ext = without_ext+"."+new_extension
    return with_new_ext



"""
Download ModelNet40 from https://modelnet.cs.princeton.edu/
Change ABS_PATH_TO_MODELNET40 depending on where it is located on you pc.
If the assert statement fails, fix the absolute path
"""

ABS_PATH_TO_MODELNET40 = "/home/ola/library/datasets/ModelNet40" 
OUTPUT_FILE_FORMAT = "ply"
OUTPUT_DATASET_DIR = "ModelNet40-norm-"+OUTPUT_FILE_FORMAT

ds_path = ABS_PATH_TO_MODELNET40
classes = os.listdir(ds_path)
assert "chair" in classes and "sink" in classes and "plant" in classes

os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)
train_type = ["test", "train"]
for classname in classes:
    for train_or_test in train_type:
        print("Processing", classname, "in", train_or_test)
        read_dir = os.path.join(ds_path, classname, train_or_test)
        read_files = [os.path.join(read_dir, filename) for filename in os.listdir(read_dir)]
        out_dir = os.path.join(OUTPUT_DATASET_DIR, classname, train_or_test)
        os.makedirs(out_dir, exist_ok=True)
        out_files = [os.path.join(out_dir, change_extension(filename, OUTPUT_FILE_FORMAT)) for filename in os.listdir(read_dir)]
        for (read_file, out_file) in zip(read_files, out_files):
            mesh = tm.load(read_file)
            rescale_mesh(mesh)
            center_mesh_to_centroid(mesh)
            #mesh = decimate_mesh(mesh, 0.3)
            tm.exchange.export.export_mesh(mesh, out_file)













