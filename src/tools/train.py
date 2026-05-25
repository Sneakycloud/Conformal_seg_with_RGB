
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

    rng = np.random.default_rng(42)

    dataset = dh.SUNRGBDDataset2(
    root=folder_path,
    num_classes=13,
    transform=transforms.Compose([
        dh.RandomCrop((256, 256)),
        dh.RandomVerticalFlip(p=0.5),
        dh.RandomHorizontalFlip(p=0.5),
        dh.RandomRotation(p=0.5)
    ]),
    img_size=(480, 640)
    )

    if class_list == None:
        class_list = dataset.class_count()
        print(f"total classes = {class_list}")
    print("Dataset loading completed")
    
    MAX_SAMPLES = 2000
    if len(dataset) > MAX_SAMPLES:
        cpu_generator = torch.Generator(device='cpu').manual_seed(seed)
        indices = torch.randperm(len(dataset), 
                                generator=cpu_generator, device='cpu')[:MAX_SAMPLES]
        dataset = torch.utils.data.Subset(dataset, indices.tolist())
        print(f'Subsampled dataset to {MAX_SAMPLES} images', flush=True)
    
    #To prepare for cross validation: https://discuss.pytorch.org/t/using-k-fold-cross-validation-to-train-my-model/196288
    
    torch.manual_seed(seed)

    torch.cuda.empty_cache()

    logging.info('Start training')
    #saves the final results from each fold for the final evaluation
    results = []
    
    splits = 10

    total_len = len(dataset)
    fold_sizes = [total_len // splits] * splits

    remainder = total_len % splits
    for r in range(remainder):
        fold_sizes[r] += 1

    split_gen = torch.Generator(device='cpu').manual_seed(seed)
    shuffled_indices = torch.randperm(total_len, generator=split_gen, device='cpu').tolist()

    folds = []
    current_idx = 0
    for fold_size in fold_sizes:
        fold_indices = shuffled_indices[current_idx : current_idx + fold_size]
        folds.append(torch.utils.data.Subset(dataset, fold_indices))
        current_idx += fold_size

    for fold in range(splits):
        logging.info(f'Fold {fold} training has started')
        print(f'Fold {fold} training has started')
        
        #in channels, how many output classes
        if mode == "RGB":
            model = unet.UNet(3, class_list, mode)
        elif mode == "D":
            model = unet.UNet(1, class_list, mode)
        elif mode == "RGBD": 
            model = unet.UNet(4, class_list, mode)
        elif mode == "TMC":
            model = unet.MultiViewFusionRGBD(classes=class_list, lamda_epochs=epochs)
        print("Model created")

        train_set = ConcatDataset([x for i,x in enumerate(folds) if i != fold])
        valid_set = folds[fold]
        print("Datasets concatenated")
        
        #  Dataloaders here
        train_it = torch.utils.data.DataLoader(
                                        train_set, shuffle=False,
                                        batch_size=batch_size, num_workers=0)
                                
        valid_it = torch.utils.data.DataLoader(
                                        valid_set, shuffle=False,
                                        batch_size=batch_size, num_workers=0)
        
        #Training
        print("Starting model fitting")
        model.fit(train_it, valid_it, epochs, log_path,None)
        print("Starting final evaluation for f1_precision, MCC, AUSE, AUCE")
        #f1 has scores for every class in format: [(class_id,f1_score)]
        #mcc has format: (mcc_score)
        #auces and auses has format: (methods, auces_scores)
        f1, mcc, auces, auses = model.final_evaluation(valid_it, class_list, log_path)

        print(f"auces variable type: {type(auces)}")
        print(f"auces raw content: {auces}")
        if isinstance(auces, (tuple, list)):
            print(f"Index 0 contains (type {type(auces[0])}): {auces[0]}")
            if len(auces) > 1:
                print(f"Index 1 contains (type {type(auces[1])}): {auces[1]}")

        fold_results = [f"Fold {fold} result", f1, mcc, auces, auses]
        
        #save results
        results.append(fold_results)
        
        
        print(f'Fold {fold} training has ended')
        
        logging.info(f'Fold {fold} training has ended')
        
        del model
        torch.cuda.empty_cache()
        
    #save results from evaluation
    pd.DataFrame(results).to_csv(os.path.join(log_path, f"Final_eval_results_{mode}"), index=False)

    mean_f1 = np.mean(np.array([[f1[1] for f1 in fold[1]] for fold in results]), axis=0) # (classes)
    mean_mcc = np.mean(np.array([fold[2] for fold in results])) # (mean_mcc)
    mean_auces = np.mean(np.array([fold[3][0] for fold in results]), axis=0) # (fold,classes->mean_auces_scores)
    mean_auses = np.mean(np.array([fold[4][0] for fold in results]), axis=0) # (fold,classes->mean_auses_scores)
    
    std_f1 = np.std(np.array([[class_tuple[1] for class_tuple in fold[1]] for fold in results]), axis=0) # (classes,mean_f1)
    std_mcc = np.std(np.array([fold[2] for fold in results])) # (mean_mcc)
    std_auces = np.std(np.array([fold[3][0] for fold in results]), axis=0) # (classes,mean_auces_scores)
    std_auses = np.std(np.array([fold[4][0] for fold in results]), axis=0) # (classes,mean_auses_scores)

    num_rows = max(len(mean_f1), len(mean_auces), len(mean_auses))
    
    rows = []
    for i in range(num_rows):

        if i < len(mean_f1):
            class_label = f"Class_{i}"
        else:
            class_label = "Global_Uncertainty"

        row = {
            "class_id": class_label,
            "mean_f1": mean_f1[i] if i < len(mean_f1) else np.nan,
            "std_f1": std_f1[i] if i < len(std_f1) else np.nan,
            "mean_auces": mean_auces[i] if i < len(mean_auces) else np.nan,
            "std_auces": std_auces[i] if i < len(std_auces) else np.nan,
            "mean_auses": mean_auses[i] if i < len(mean_auses) else np.nan,
            "std_auses": std_auses[i] if i < len(std_auses) else np.nan,
        }
        rows.append(row)

    print(f"\n--- DEBUGGING COLUMN SHAPES ---")
    
    try: print(f"Total structured rows to save: {len(rows)}")
    except Exception as e: print(f"rows error: {e}")
        
    try: print(f"mean_f1 shape: {mean_f1.shape if hasattr(mean_f1, 'shape') else len(mean_f1)}")
    except Exception as e: print(f"mean_f1 error: {e}")
    try: print(f"std_f1 shape: {std_f1.shape if hasattr(std_f1, 'shape') else len(std_f1)}")
    except Exception as e: print(f"std_f1 error: {e}")
        
    try: print(f"mean_auces shape: {mean_auces.shape if hasattr(mean_auces, 'shape') else len(mean_auces)}")
    except Exception as e: print(f"mean_auces error: {e}")
    try: print(f"std_auces shape: {std_auces.shape if hasattr(std_auces, 'shape') else len(std_auces)}")
    except Exception as e: print(f"std_auces error: {e}")
        
    try: print(f"mean_auses shape: {mean_auses.shape if hasattr(mean_auses, 'shape') else len(mean_auses)}")
    except Exception as e: print(f"mean_auses error: {e}")
    try: print(f"std_auses shape: {std_auses.shape if hasattr(std_auses, 'shape') else len(std_auses)}")
    except Exception as e: print(f"std_auses error: {e}")
        
    try: 
        mcc_mean_val = "Scalar Value" if np.isscalar(mean_mcc) else (mean_mcc.shape if hasattr(mean_mcc, 'shape') else len(mean_mcc))
        print(f"mean_mcc representation: {mcc_mean_val}")
    except Exception as e: print(f"mean_mcc error: {e}")
        
    try: 
        mcc_std_val = "Scalar Value" if np.isscalar(std_mcc) else (std_mcc.shape if hasattr(std_mcc, 'shape') else len(std_mcc))
        print(f"std_mcc representation: {mcc_std_val}")
    except Exception as e: print(f"std_mcc error: {e}")
        
    print(f"--------------------------------\n")
    
    pd.DataFrame(rows).to_csv(
        os.path.join(log_path, f"Final_class_metrics_{mode}.csv"), index=False
    )
    
    global_results_dict = {
        "metric": ["MCC"],
        "mean": [mean_mcc],
        "std": [std_mcc]
    }
    pd.DataFrame(global_results_dict).to_csv(
        os.path.join(log_path, f"Final_global_summary_{mode}.csv"), index=False
    )
    
    print("All evaluation metrics saved successfully without dimension conflicts!")
    logging.info('End training')
