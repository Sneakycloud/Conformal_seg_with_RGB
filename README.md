# Conformal semantic segmentation for RGB and depth utilizing Dirichlet distributions
Forked from the repository accompanying the paper "Uncertainty Quantification for LiDAR-based Maps of Ditches and Natural Streams".

The modifications we made:
- Made dataloaders for [reorganized SUN RGB-D](https://github.com/chrischoy/SUN_RGBD/tree/master) and [NYU depth v2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) for semantic segmentation using rgb and depth
- Data exploration for SUN RGB-D
- Modified U-net to use rgb and depth
- Created the trusted multi-view ([TMC](https://arxiv.org/abs/2204.11423)) based model for semantic segmentation
- Vector optimized loss for semantic segmentation for TMC
- Modified metric calculations to work with new data loader
- Implemented cross-validitation
- Reduced memory usage

## Source Code

The source code and instructions on how to install and run the provided
implementation can be found in [src](src/).


## Experiment Data

The raw data collected through the conducted experiments can be found in [logs](logs/).


## Groundtruth Data

The reorganized SUN RGBD dataset that was used can be retrieved from [here](https://github.com/chrischoy/SUN_RGBD/tree/master)
