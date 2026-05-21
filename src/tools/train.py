
import logging
import numpy as np
import torch
import pandas as pd
import os
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
def rgbd_train(folder_path,log_path,seed,epochs,batch_size, class_list = None, mode = "RGBD"):
    '''
    Trains a 4 input channel rgb-d unet on nyu v2
    
    :param str folder_path: The filepath to a .mat containing "image", "depth", and "label"
    :param str log_path: where the logs should be
    :param int seed: for reproducability
    :param int epochs: number of epochs to train
    :param int batch_size: how many images in each batch
    :param [int] class_list: list of all class ints for target classes
    :param string mode: Describes which models to create. Either RGB, D, or RGBD.
    '''

    print("Dataset loading started")
    util.enable_logging(log_path, 'train.log')
    rng = np.random.default_rng(seed)
    dataset = dh.RGBD_Segmentation_Dataset(folder_path, transforms.Compose([dh.RandomHorizontalFlip(),dh.RandomVerticalFlip(), dh.RandomRotation()]))
    if class_list == None:
        class_list = dataset.class_count()
        print(f"total classes = {class_list}")
    print("Dataset loading completed")
    
    #To prepare for cross validation: https://discuss.pytorch.org/t/using-k-fold-cross-validation-to-train-my-model/196288
    
    torch.manual_seed(seed)

    
    logging.info('Start training')
    #saves the final results from each fold for the final evaluation
    results = []
    
    
    
    splits = 10
    split_range = [1 / splits for _ in range(splits)]
    folds = random_split(dataset, split_range,generator=torch.Generator('cpu').manual_seed(seed))
    for fold in range(splits):
        logging.info(f'Fold {fold} training has started')
        print(f'Fold {fold} training has started')
        
        #in channels, how many output classes
        if mode == "RGB":
            model = unet.UNet(3, class_list, "RGB")
        elif mode == "D":
            model = unet.UNet(1, class_list, "D")
        elif mode == "RGBD": 
            model = unet.UNet(4, class_list, "RGBD")
        elif mode == "TMC":
            model = unet.MultiViewFusionRGBD(class_list)
        print("Model created")

        train_set = ConcatDataset([x for i,x in enumerate(folds) if i != fold])
        valid_set = folds[fold]
        print("Datasets concatenated")
        
        #  Dataloaders here
        train_it = torch.utils.data.DataLoader(
                                train_set, shuffle=False,
                                batch_size=batch_size, num_workers=0,
                                generator=torch.Generator('cuda')
                                                .manual_seed(seed))
        valid_it = torch.utils.data.DataLoader(
                                        valid_set, shuffle=False,
                                        batch_size=batch_size, num_workers=0,
                                        generator=torch.Generator('cuda')
                                                    .manual_seed(seed))
        
        #Training
        print("Starting model fitting")
        model.fit(train_it, valid_it, epochs, log_path,None)
        print("Starting final evaluation for f1_precision, MCC, AUSE, AUCE")
        #f1 has scores for every class in format: [(class_id,f1_score)]
        #mcc has format: (mcc_score)
        #auces and auses has format: (methods, auces_scores)
        f1, mcc, auces, auses = model.final_evaluation(valid_it, class_list, log_path)
        fold_results = [f"Fold {fold} result", f1, mcc, auces, auses]
        
        #save results
        results.append(fold_results)
        
        
        print(f'Fold {fold} training has ended')
        
        logging.info(f'Fold {fold} training has ended')
        
        
    #save results from evaluation
    pd.DataFrame(results).to_csv(os.path.join(log_path, f"Final_eval_results_{mode}"), index=False)
    
    
    classes_id = [fold[1][0] for fold in results]
    mean_f1 = np.mean(np.array([[class_tuple[1] for class_tuple in fold[1]] for fold in results]), axis=1) # (classes,mean_f1)
    mean_mcc = np.mean(np.array([fold[2] for fold in results]), axis=0) # (mean_mcc)
    mean_auces = np.mean(np.array([fold[3][0] for fold in results]), axis=1) # (classes,mean_auces_scores)
    mean_auses = np.mean(np.array([fold[4][0] for fold in results]), axis=1) # (classes,mean_auses_scores)
    
    std_f1 = np.std(np.array([[class_tuple[1] for class_tuple in fold[1]] for fold in results]), axis=1) # (classes,mean_f1)
    std_mcc = np.std(np.array([fold[2] for fold in results]), axis=0) # (mean_mcc)
    std_auces = np.std(np.array([fold[3][0] for fold in results]), axis=1) # (classes,mean_auces_scores)
    std_auses = np.std(np.array([fold[4][0] for fold in results]), axis=1) # (classes,mean_auses_scores)
    
    mean_std_results = {
        "classes":classes_id,
        "mean_f1":mean_f1,"mean_mcc":mean_mcc,"mean_auces":mean_auces,"mean_auses":mean_auses,
        "std_f1":std_f1,"std_mcc":std_mcc,"std_auces":std_auces,"std_auses":std_auses
        }
    
    pd.DataFrame(mean_std_results).to_csv(os.path.join(log_path, f"Final_eval_results_mean_std_{mode}"), index=False)
    
    logging.info('End training')
    
    
