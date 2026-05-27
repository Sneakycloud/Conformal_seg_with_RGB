# Conformal semantic segmentation for RGB and depth utilizing Dirichlet distributions
Forked from the repository accompanying the paper "Uncertainty Quantification for LiDAR-based Maps of Ditches and Natural Streams".

The modifications we made:
- Made rgb,depth,and mask dataloaders for SUN-RGBD and NYU depth v2 for semantic segmentation
- Data exploration for SUN-RGBD
- Modified U-net to use rgb and depth
- Created the trusted multi-view (TMC) model for semantic segmentation
- Vector optimized loss for semantic segmentation for TMC
- Modified metric calculations to work with new data loader
- Implemented cross-validitation
- Reduced memory usage

## Source Code

The source code and instructions on how to install and run the provided
implementation can be found in [src](src/).


## Experiment Data

The raw data collected through the conducted experiments and its description
can be found in [data](data/).


## Groundtruth Data

The reorganized SUN RGBD dataset that was used can be retrieved from [here](https://github.com/chrischoy/SUN_RGBD/tree/master)
