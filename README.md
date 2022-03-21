# A baseline for end-to-end RGB render-and-compare pose estimation
A baseline for render-and-compare machine learning pose estimation using a known CAD model, without the use of depth measurements.

The work is inspired by the DeepIM paper and CosyPose paper, but this repository contains a simplified version.
Check them out here:

**DeepIM**:
https://arxiv.org/abs/1804.00175

**CosyPose**:
https://arxiv.org/abs/2008.08465
https://github.com/ylabbe/cosypose

# How to run the baseline
1. Download ModelNet40 from https://modelnet.cs.princeton.edu/
2. Change the dataset path in **normalize_modelnet40.py** and run the file
3. You should now have a directory named "ModelNet40-norm-ply" in project root directory



