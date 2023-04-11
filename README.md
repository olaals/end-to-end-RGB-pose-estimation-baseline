# A baseline for end-to-end RGB render-and-compare pose estimation

![Alt text](docs/example.png "Training inference process")



A baseline for render-and-compare machine learning pose estimation using a known CAD model, without the use of depth measurements.


\
Links to DeepIM and CosyPose:

**DeepIM**:
https://arxiv.org/abs/1804.00175


**CosyPose**:
https://arxiv.org/abs/2008.08465
\
https://github.com/ylabbe/cosypose

Snippets of code are copied from the CosyPose github. Copied functions contains an explicit comment about the source.

# Pre-requisites
## 1: Install dependencies
1. Install torch and cuda from https://pytorch.org/get-started/locally/
2. ```pip install -r requirements.txt```

## 2. Image dataset
1. Create an image dataset from https://github.com/olaals/datasets-rgb-pose-estimation 
2. Create a symbolic link or copy the dataset to ![img-datasets](img-datasets)

## 3. 3D dataset
1. Create a symbolic link or copy the same 3D model dataset used to create the image dataset to ![model3d-datasets](model3d-datasets)

# Training and testing
## 1: Create a config file
1. Create a config file in ![configs][configs] 
2. An example config file is given in ![example_config.py][configs/example_config.py]
## 2: Training a model
1. To train a model, run the following command
```bash
python train_model.py configs/example_config.py
```





# Technical details
## Training process overview
The exact training process depends on the configuration that is set in the config files in the configs directory, but
the overall pipeline is shown below

![Alt text](irrelevant-data/training-inference-process.png "Training inference process")

The general pipeline includes
- A renderer that produces two images of the same object, where the initial guess of the object is slightly off.
- These images are concatenated and used as the input to a convolutional neural network.
- The CNN tries to estimate either a 6D or 9D representation of rotation, and pixel translation in x and y direction, as well as a depth parameter vz.
- The output of the CNN is passed onto a rotation representation function, which calculates a valid rotation matrix
- The pixel translation output of the CNN is converted to translation in Euclidean space.
- Together, the rotation matrix and translation form a transformation matrix delta_T, which updates the current estimate of T_CO with T_CO_new = delta_T*T_CO
- A loss function loss(T_CO_new, T_CO_gt) determines a number which represents the deviation between T_CO_new and T_CO_gt

## Frames
The code in the repository uses shorthand notation for the transformation 
matrix describing the rotation and translation between frames. The image
below shows the shorthand notations used, where T_CO is of particular importance.

![Alt text](docs/scene-frames.png "Scene frames")


