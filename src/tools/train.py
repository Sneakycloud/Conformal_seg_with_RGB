
import logging
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split
from sklearn.model_selection import KFold

import utils.data_handling as dh
import utils.unet as unet
import utils.util as util


def main(img_path, gt_path, log_path, train_ids, valid_ids, seed, epochs,
         batch_size, classes, weighting):
    ''' Train model '''

    util.enable_logging(log_path, 'train.log')
    rng = np.random.default_rng(seed)
    train_set = dh.MultibandDataset(img_path, classes, train_ids,
                                    gt_path=gt_path,
                                    transform=transforms.Compose([
                                            dh.RandomFlip(rng),
                                            dh.RandomRotate(rng),
                                            dh.ToTensor(),
                                            dh.ToOnehotGaussianBlur(7,
                                                                    classes,
                                                                    False)]))
    valid_set = dh.MultibandDataset(img_path, classes, valid_ids,
                                    gt_path=gt_path,
                                    transform=transforms.Compose([
                                            dh.ToTensor(),
                                            dh.ToOnehotGaussianBlur(7,
                                                                    classes,
                                                                    False)]))
    torch.manual_seed(seed)
    train_it = torch.utils.data.DataLoader(
                                    train_set, shuffle=True,
                                    batch_size=batch_size, num_workers=0,
                                    generator=torch.Generator('cuda')
                                                   .manual_seed(seed))
    valid_it = torch.utils.data.DataLoader(
                                    valid_set, shuffle=True,
                                    batch_size=batch_size, num_workers=0,
                                    generator=torch.Generator('cuda')
                                                   .manual_seed(seed))

    model = unet.UNet(len(img_path),
                      dh.MultibandDataset.parse_classes(classes))
    logging.info('Start training')
    model.fit(train_it, valid_it, epochs, log_path,
              train_set.infer_weights(weighting))
    logging.info('End training')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                        description='Train Model',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-I', '--img_path', action='append', help='Add path '
                        'to input images')
    parser.add_argument('gt_path', help='Path to groundtruth image folder')
    parser.add_argument('log_path', help='Path to folder where logging data '
                        'will be stored')
    parser.add_argument('train_ids', help='File containing image names for '
                        'training')
    parser.add_argument('valid_ids', help='File containing image names for '
                        'validation')
    parser.add_argument('--seed', help='Random seed', default=None, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', help='Number of patches per batch',
                        type=int, default=4)
    parser.add_argument('--classes', help='List of class labels in ground '
                        'truth - order needs to correspond to weighting order',
                        default='0,1,2')
    parser.add_argument('--weighting', help='Configure class weights - can be '
                        '"mfb", "none", or defined weight string, '
                        'e.g., "0.1,1,1"', default='mfb')

    args = vars(parser.parse_args())
    main(**args)

#Trains a 4 input channel unet
def rgbd_train(folder_path,log_path,seed,epochs,batch_size, class_list = None):
    '''
    Trains a 4 input channel rgb-d unet on nyu v2
    
    :param str folder_path: The filepath to a .mat containing "image", "depth", and "label"
    :param str log_path: where the logs should be
    :param int seed: for reproducability
    :param int epochs: number of epochs to train
    :param int batch_size: how many images in each batch
    :param [int] class_list: list of all class ints for target classes
    '''

    print("Dataset loading started")
    util.enable_logging(log_path, 'train.log')
    rng = np.random.default_rng(seed)
    dataset = dh.RGBD_Segmentation_Dataset(folder_path, transforms.Compose([dh.RandomHorizontalFlip(),dh.RandomVerticalFlip()]))
    if class_list == None:
        class_list = dataset.class_count()
        print(f"total classes = {class_list}")
    print("Dataset loading completed")
    
    #To prepare for cross validation: https://discuss.pytorch.org/t/using-k-fold-cross-validation-to-train-my-model/196288
    
    torch.manual_seed(seed)

    #in channels, how many output classes
    model = unet.UNet(4, class_list)
    print("Model created")
    logging.info('Start training')
    
    splits = 10
    split_range = [1 / splits for _ in range(splits)]
    
    folds = random_split(dataset, split_range,generator=torch.Generator('cuda').manual_seed(seed))
    for fold in range(splits):
        train_set = folds[fold]
        valid_set = ConcatDataset([x for i,x in enumerate(folds) if i != fold])
        
        #  Dataloaders here
        #  Example
        logging.info(f'Fold {fold} training has started')
        print(f'Fold {fold} training has started')
        train_it = torch.utils.data.DataLoader(
                                train_set, shuffle=False,
                                batch_size=batch_size, num_workers=2,
                                generator=torch.Generator('cuda')
                                                .manual_seed(seed))
        valid_it = torch.utils.data.DataLoader(
                                        valid_set, shuffle=False,
                                        batch_size=batch_size, num_workers=2,
                                        generator=torch.Generator('cuda')
                                                    .manual_seed(seed))
        
        model.fit(train_it, valid_it, epochs, log_path,None)
        print(f'Fold {fold} training has ended')
        logging.info(f'Fold {fold} training has ended')
    
    logging.info('End training')
