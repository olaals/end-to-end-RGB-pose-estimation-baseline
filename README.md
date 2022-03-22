# A baseline for end-to-end RGB render-and-compare pose estimation
A baseline for render-and-compare machine learning pose estimation using a known CAD model, without the use of depth measurements.

The work is inspired by the DeepIM paper and CosyPose paper, but this repository contains a simplified version.
\\
The simplifications includes:
- No handling of symmetries
- Simpler renderer compared to CosyPose
- No disentangled loss (see CosyPose)
- No auxiliary training path (DeepIM)
- Baseline network is simple

\\
Links to DeepIM and CosyPose:

**DeepIM**:
https://arxiv.org/abs/1804.00175

**CosyPose**:
https://arxiv.org/abs/2008.08465
https://github.com/ylabbe/cosypose

Snippets of code are copied from the CosyPose github. Copied functions contains an explicit comment about the source.

# 1: Install dependencies
1. Install torch and cuda from https://pytorch.org/get-started/locally/
2. ```pip install -r requirements.txt```



# 2: Run the baseline
1. Download ModelNet40 from https://modelnet.cs.princeton.edu/
2. Change the dataset path in **normalize_modelnet40.py** and run the file
3. You should now have a directory named "ModelNet40-norm-ply" in project root directory


# 3: Suggestions for things to test
- Compare baseline network with EfficientNet (used in CosyPose)
- Compare 6D representation of rotation with 9D represenat(SVD)  
- Test multiple feed forward heads on the baseline networks, for individually predicting rotation, horisontal and vertical translation, and depth
- Test different activation functions on these heads
- Change the classes in ModelNet40 the network is trained on, and see if it generalizes well to unseen classes (i.e. remove the "chair" class from training, and test it on the "chair" class after training on other objects)




**License**:
I included an MIT license, but feel free to copy any code without citing this repository. If you copy any code which stems from another repository in this repository, read their specific license and cite accordingly.
