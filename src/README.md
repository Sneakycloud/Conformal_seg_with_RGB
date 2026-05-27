# Source Code

## Installation

Versions of installed packages are in [requirements.txt](requirements.txt).

Alternatively the following commands should work:
- pip install numpy h5py matplotlib torchvision tqdm pandas torch scikit-learn tifffile
- pip install concretedropout
- pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# To run
First install the [reorganized SUN RGB-D](https://github.com/chrischoy/SUN_RGBD/tree/master) and follow the instructions therein.

Secondly change the `datapath` variable in controller.py to whatever folder you have installed your data in.

Lastly use ``python controller.py`` in the [tools](tools/) directory.

If desired then parameters can be changed in the controller.py:
- Epochs = 50
- Batch_size = 16
- Mode = "RGB", "D", "RGBD" or "TMC". Where each corresponds to their specific model as outlined in our study.
- Seed = 42. This was for consistent testing but may be adjusted as required

There are some other parameters that may want to be adjusted based on the use_case in rgbd_train() in ``train.py`` which is also located in [tools](tools/) directory.
- num_classes = 13 or 37 (This automatically uses either the 13 or 37 class SUN RGB-D)
- Used transforms for the data augmentation. Of particular intresst is the random crop size and to what size the image will be resized to.
- MAX_SAMPLES = 0 < x < 10335. This argument restricts how many samples will be selected from the SUN RGB-D dataset
- splits = 10. This decides how many k-fold cross validation is done

The results from running ``controller.py`` is by default saved in the [logs](../logs) directory, and includes:
- Average MCC across all folds.
- Mean and std for F1 score, AUSE, and AUCE for all classes
