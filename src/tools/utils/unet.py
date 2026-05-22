import math
import os

import concretedropout.pytorch as cd
import numpy as np
import torch
import evaluate
import pandas as pd
from torch import nn
from tqdm import tqdm
from ..evaluate_uncertainty_maps import evaluate_uncertainty
from torch.nn import ReLU
import torch.nn.functional as F

#from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm


class IoU:

    '''Class-wise intersecion over union metric'''

    def __init__(self, classes, regression=False):
        '''Setup IoU metric

        Parameters
        ----------
        classes : List of classes (int)
        regression : Compute metric for regression problem
        '''
        self._regression = regression
        self._results = {f: [] for f in classes}

    def reset(self):
        '''Reset internal datastructure
        '''
        for key in self._results:
            self._results[key] = []

    def add(self, predicted, target):
        '''Add result for predicted and target pair to metric

        Parameters
        ----------
        predicted : Tensor of predicted segmentation
        target : Tensor of targets

        '''
        # undo log transform (may not be necessary)
        if self._regression:
            predicted = torch.exp(-torch.exp(predicted))
            target = torch.exp(-torch.exp(target))

        predicted = (predicted.argmax(dim=1)
                     .view(-1))
        target = (target.argmax(dim=1)
                  .view(-1))

        # If target and/or predicted are tensors, convert them to numpy arrays
        if torch.is_tensor(predicted):
            predicted = predicted.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()

        for label in self._results:
            pred_label = predicted == label
            target_label = target == label

            intersect = np.logical_and(pred_label, target_label).sum()
            union = np.logical_or(pred_label, target_label).sum()

            if union > 0:
                self._results[label].append(intersect/union)

    def value(self):
        '''Compute class-wise IoU
        Returns
        -------
        Class - IoU map

        '''
        results = {}

        for label in self._results:
            results[label] = np.mean(self._results[label])

        return results


class WeightedMSE(nn.Module):

    '''Weighted mean square error loss'''

    def __init__(self, weights=None, device='cuda'):
        '''Setup loss

        Parameters
        ----------
        weights : List of class weights, optional
        device : Select device to run on, optional
        '''
        nn.Module.__init__(self)
        if weights != None:
            self._weights = torch.from_numpy(weights).to(device)
        else:
            self._weights = None

    def forward(self, inputs, targets):
        '''Compute loss for given inputs and targets

        Parameters
        ----------
        inputs : Predicted output
        targets : Target output

        Returns
        -------
        Weighted MSE

        '''
        if self._weights is not None:
            mse = ((inputs - targets) ** 2).mean(dim=(0, 2, 3))
            mse = (self._weights * mse).sum()
        else:
            mse = ((inputs - targets) ** 2).mean()

        return mse


class ModelCheckpoint:

    '''Save best performing model'''

    def __init__(self, log_path, model, optimizer,
                 file_name='minloss_checkpoint.pth.tar'):
        '''Setup checkpoiting

        Parameters
        ----------
        log_path : Path to store model file to
        model : Model reference
        optimizer : Optimizer reference
        file_name : Name of the checkpoint

        '''
        self._model_path = os.path.join(log_path, file_name)
        self._model = model
        self._optimizer = optimizer
        self._minloss = np.inf

    def update(self, epoch, loss):
        '''Save model if given loss improved

        Parameters
        ----------
        epoch : Current epoch number
        loss : New loss value

        '''
        if loss < self._minloss:
            self._minloss = loss
            state = {'epoch': epoch, 'state_dict': self._model.state_dict(),
                     'opt_dict': self._optimizer.state_dict()}
            torch.save(state, self._model_path)


class CSVLogger:

    '''Write training log'''

    def __init__(self, log_path, metric, file_name='log.csv'):
        '''Setup logging

        Parameters
        ----------
        log_path : Path to store CSV log
        metric : Metric to log
        file_name : Name of the log file

        '''
        self._file = os.path.join(log_path, file_name)
        self._metric_order = [f for f in metric.value()]

        columns = ['epoch', 'loss']
        columns += [f'iou_{f}' for f in self._metric_order]
        columns += ['lr', 'val_loss']
        columns += [f'val_iou_{f}' for f in self._metric_order]

        with open(self._file, 'w', encoding='utf-8') as log:
            log.write(','.join(columns) + '\n')

    def update(self, epoch, train_loss, train_metric, valid_loss, valid_metric,
               learning_rate):
        '''Write current progress

        Parameters
        ----------
        epoch : Current epoch number
        train_loss : Training loss
        train_metric : Training metric results
        valid_loss : Validation loss
        valid_metric : Validation metric results
        learning_rate : Current learning rate

        '''
        entry = [epoch, train_loss]
        entry += [train_metric[f] for f in self._metric_order]
        entry += [learning_rate, valid_loss]
        entry += [valid_metric[f] for f in self._metric_order]

        with open(self._file, 'a', encoding='utf-8') as log:
            log_entry = ','.join(str(el) for el in entry)
            log.write(log_entry + '\n')


class BaseUNet(nn.Module):

    '''UNet architecture class'''

    LAYER_CONFIG = [32, 64, 128, 256]

    @classmethod
    def _pad(cls, inputs):
        '''Add padding if needed

        Parameters
        ----------
        inputs : Input tensor

        Returns
        -------
        Padded tensor, applied padding

        '''
        _, _, height, width = inputs.size()

        width_correct = 2**math.ceil(math.log2(width)) - width
        height_correct = 2**math.ceil(math.log2(height)) - height

        left = width_correct // 2
        right = width_correct - left
        top = height_correct // 2
        bottom = height_correct - top

        padding = (top, bottom, left, right)
        return nn.functional.pad(inputs, (left, right, top, bottom)), padding

    @classmethod
    def _unpad(cls, inputs, padding):
        '''TODO: Docstring for _unpad.

        Parameters
        ----------
        inputs : Input tensor
        padding : Padding information

        Returns
        -------
        Tensor without padding

        '''
        top, bottom, left, right = padding

        bottom = -1 if bottom == 0 else -bottom
        right = -1 if right == 0 else -right

        return inputs[:, :, top:bottom, left:right]

    @classmethod
    def _downsample(cls, inputs, layers, act, pool, steps_per_layer=3):
        '''Downsample inputs

        Parameters
        ----------
        inputs : Input tensor
        layers : Layer steps
        act : Activation function to use
        pool : Pooling function to use
        steps_per_layer : Computation steps per layer, optional

        Returns
        -------
        Output tensor, skip tensors

        '''
        result = inputs
        skips = []

        layer_num = len(layers) // steps_per_layer
        for layer_i in range(layer_num):
            conv1 = layers[layer_i*steps_per_layer]
            drop = layers[layer_i*steps_per_layer+1]
            conv2 = layers[layer_i*steps_per_layer+2]

            result = drop(result, conv1)
            result = act(result)
            result = conv2(result)
            result = act(result)
            skips.append(result)
            result = pool(result)

        return result, skips

    @classmethod
    def _process(cls, inputs, layers, act):
        '''Implement lowest level in UNet

        Parameters
        ----------
        inputs : Input tensor
        layers : Layer steps
        act : Activation function to use

        Returns
        -------
        Output tensor

        '''
        conv1 = layers[0]
        drop = layers[1]
        conv2 = layers[2]

        result = drop(inputs, conv1)
        result = act(result)
        result = conv2(result)
        result = act(result)

        return result

    @classmethod
    def _upsample(cls, inputs, skips, layers, act, steps_per_layer=4):
        '''Upsample inputs

        Parameters
        ----------
        inputs : Input tensor
        skips : Skip tensors
        layers : Layer steps
        act : Activation function to use
        steps_per_layer : Computation steps per layer, optional

        Returns
        -------
        Output tensor

        '''
        # auto_LiRPA seems to sometimes run inputs on CPU and sometimes on GPU,
        # which collides with the skip tensors
        if not inputs.is_cuda:
            tmp = []
            for skip in skips:
                tmp.append(skip.cpu())
            skips = tmp
        result = inputs

        layer_num = len(layers) // steps_per_layer
        for layer_i in range(layer_num):
            tconv = layers[layer_i*steps_per_layer]
            conv1 = layers[layer_i*steps_per_layer+1]
            drop = layers[layer_i*steps_per_layer+2]
            conv2 = layers[layer_i*steps_per_layer+3]

            result = tconv(result)
            # result = torch.cat((result, skips[-1 * (layer_i+1)]), dim=1)
            result = torch.cat((result, skips.pop()), dim=1)
            result = drop(result, conv1)
            result = act(result)
            result = conv2(result)
            result = act(result)

        return result


class UNet(BaseUNet):

    '''UNet implementation for uncertainty quantification using feature
    conformal prediction and concrete dropout'''

    def __init__(self, in_channels, classes, mode, device='cuda'):
        '''Setup model

        Parameters
        ----------
        in_channels : Number of input channels
        classes : List of class labels
        device : Device to run model on [cuda|cpu]
        mode: RGB, D, or RGBD
        '''
        BaseUNet.__init__(self)

        # ensure that all tensors are created on the same device
        # torch.set_default_device(device)
        if device == 'cuda':
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        self._classes = classes
        self._device = device
        self._downsampling = nn.ModuleList()
        self._processing = nn.ModuleList()
        self._upsampling = nn.ModuleList()
        self._act = nn.ReLU()
        self._pool = nn.MaxPool2d(2)
        self._mode = mode

        # downsampling
        for filters in self.LAYER_CONFIG:
            self._downsampling.append(nn.Conv2d(in_channels, filters, 3,
                                      padding='same'))
            in_channels = filters
            self._downsampling.append(cd.ConcreteDropout2D())
            self._downsampling.append(nn.Conv2d(in_channels, filters, 3,
                                      padding='same'))

        # processing
        self._processing.append(nn.Conv2d(in_channels, 512, 3, padding='same'))
        in_channels = 512
        self._processing.append(cd.ConcreteDropout2D())
        self._processing.append(nn.Conv2d(in_channels, 512, 3, padding='same'))

        # upsampling
        for filters in reversed(self.LAYER_CONFIG):
            self._upsampling.append(nn.ConvTranspose2d(in_channels, filters, 2,
                                    stride=2))
            in_channels = filters
            self._upsampling.append(nn.Conv2d(2*in_channels, filters, 3,
                                    padding=1))
            self._upsampling.append(cd.ConcreteDropout2D())
            self._upsampling.append(nn.Conv2d(in_channels, filters, 3,
                                    padding=1))

        # output
        self._output = nn.Conv2d(in_channels, len(self._classes), 1)

        # initialize conv layers
        self.apply(self._init_weights)
        self.to(device)

    def _init_weights(self, layer):
        '''Initialize weights

        Parameters
        ----------
        layer : Layer to initialize

        '''
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal(layer.weight)
            nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.ConvTranspose2d):
            nn.init.zeros_(layer.bias)

    def forward(self, inputs):
        '''Run model on given input

        Parameters
        ----------
        inputs : Input tensor

        Returns
        -------
        Predicted segmentation

        '''
        result, padding = self._pad(inputs)

        # downsampling
        result, skips = self._downsample(result, self._downsampling,
                                         self._act, self._pool)

        # processing
        result = self._process(result, self._processing, self._act)

        # upsampling
        result = self._upsample(result, skips, self._upsampling, self._act)

        result = self._output(result)

        return self._unpad(result, padding)

    def load(self, model_path):
        '''Load model weights from given path

        Parameters
        ----------
        model_path : Path to the model

        '''
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint['state_dict'])

    def fit(self, train_it, valid_it, epochs, log_dir, weights):
        '''Train model

        Parameters
        ----------
        train_it : Training data
        valid_it : Validation data
        epochs : Number of epochs to train
        log_dir : Folder to which logging information is saved
        weights : List of class weights

        '''

        criterion = WeightedMSE(weights)
        # criterion = torch.nn.CrossEntropyLoss(torch.from_numpy(
        #                                             np.array(weights))
        #                                       .to(self._device))
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        metric = IoU(self._classes, True)
        checkpoint = ModelCheckpoint(log_dir, self, optimizer)
        csv_log = CSVLogger(log_dir, metric)

        for epoch in tqdm(range(epochs)):

            train_loss, train_iou = self._train(train_it, criterion,
                                                optimizer, metric)
            scheduler.step(train_loss)

            valid_loss, valid_iou = self._validate(valid_it, criterion, metric)
            checkpoint.update(epoch, valid_loss)
            csv_log.update(epoch, train_loss, train_iou, valid_loss, valid_iou,
                           optimizer.param_groups[0]['lr'])

    def proba(self, batch):
        '''Predict class probabilities for given batch

        Parameters
        ----------
        batch : Input images

        Returns
        -------
        Probability map

        '''
        self.eval()
        if isinstance(batch, dict):
            batch = batch['inputs'].to(self._device,
                                       dtype=batch['inputs'].dtype)
        predicted = self(batch)
        
        #predicted = torch.exp(-torch.exp(predicted))

        softmax = torch.nn.Softmax2d()
        predicted = softmax(predicted)

        return predicted.detach().cpu().numpy()

    def predict(self, batch):
        '''Predict classes for given batch

        Parameters
        ----------
        batch : Input images

        Returns
        -------
        Class map

        '''
        predicted = self.proba(batch)

        return predicted.argmax(axis=1)

    def _set_mc_dropout(self, enable):
        '''Configure model to enable MC Dropout

        Parameters
        ----------
        enable : Set to True or False

        '''
        for module in filter(lambda x: isinstance(x, cd.ConcreteDropout2D),
                             self.modules()):
            module.is_mc_dropout = enable

    def mc_dropout(self, batch, samples, batch_size):
        '''Perform Monte Carlo Dropout to infer map of predictive entropy and
        mutual information

        Parameters
        ----------
        batch : Input images
        samples : Number of Monte Carlo samples
        batch_size : Maximum batch size to process at once

        Returns
        -------
        Predictive entropy map, Mutual information map

        '''
        self._set_mc_dropout(True)
        image = batch['inputs'].to(self._device, dtype=batch['inputs'].dtype)
        assert image.shape[0] == 1, 'Process one image at a time'

        batch = image.repeat(batch_size, 1, 1, 1)

        repeats = samples // batch_size
        outputs = []
        for _ in range(repeats):
            # predict
            predicted = self.proba(batch)
            outputs.append(predicted)

        pred_samples = np.concatenate(outputs)
        pred_distr = pred_samples.mean(axis=0)
        sample_size = pred_samples.shape[0]

        # calculate predictive entropy
        eps = np.finfo(pred_distr.dtype).tiny
        log_distr = np.log(pred_distr + eps)
        pred_entropy = -1 * np.sum(pred_distr * log_distr, axis=0)

        # calucalte mutual information
        log_samples = np.log(pred_samples + eps)
        minus_e = np.sum(pred_samples * log_samples, axis=(0, 1))
        minus_e /= sample_size
        mutual_info = pred_entropy + minus_e

        self._set_mc_dropout(False)

        return pred_entropy, mutual_info

    def _train(self, train_it, criterion, optimizer, metric):
        '''Train model on epoch

        Parameters
        ----------
        train_it : Training data
        criterion : Training criterion
        optimizer : Optimizer
        metric : Metric

        Returns
        -------
        Training loss, metric output

        '''
        self.train()
        epoch_loss = 0.0
        metric.reset()
        
        for (rgb, depth, target) in train_it:
            target = target.to(self._device)

            if self._mode == "RGB":
                inputs = rgb.to(self._device)
            elif self._mode == "D":
                inputs = depth.to(self._device)
            elif self._mode == "RGBD":
                inputs = torch.cat([rgb,depth],dim=1).to(self._device)

            outputs = self(inputs)

            # add dropout regularization
            reg = torch.zeros(1)  # get the regularization term
            for module in filter(lambda x: isinstance(x, cd.ConcreteDropout2D),
                                 self.modules()):
                reg += module.regularization
            loss = criterion(outputs, target) + reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            metric.add(outputs.detach(), target.detach())

        return epoch_loss/len(train_it), metric.value()

    def _validate(self, valid_it, criterion, metric):
        '''Validate model

        Parameters
        ----------
        valid_it : Validation data
        criterion : Training criterion
        metric : Metric

        Returns
        -------
        Validation loss, validation metric

        '''
        self.eval()
        epoch_loss = 0.0
        metric.reset()

        for batch_idx, (rgb, depth, target) in enumerate(valid_it):
            target = target.to(self._device)

            if self._mode == "RGB":
                inputs = rgb.to(self._device)
            elif self._mode == "D":
                inputs = depth.to(self._device)
            elif self._mode == "RGBD":
                inputs = torch.cat([rgb,depth],dim=1).to(self._device)

            with torch.no_grad():
                outputs = self(inputs)
                loss = criterion(outputs, target)

            epoch_loss += loss.item()
            metric.add(outputs.detach(), target.detach())

        return epoch_loss/len(valid_it), metric.value()

    def final_evaluation(self, valid_it, classes, log_path):
        self.eval()

        #Prep for eval
        results = {'img': [], 'mcc': []}
        confusion = {f: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for f in classes}
        predictions = torch.zeros((len(valid_it),valid_it.shape[0],valid_it.shape[1],valid_it.shape[2],valid_it.shape[3])) # (Batch,batch_size,channel,height,width)
        for cls_id in classes:
            results[f'f_{cls_id}'] = []


        for batch_idx, (rgb, depth, target) in enumerate(valid_it):
            target = target.to(self._device)

            if self._mode == "RGB":
                inputs = rgb.to(self._device)
            elif self._mode == "D":
                inputs = depth.to(self._device)
            elif self._mode == "RGBD":
                inputs = torch.cat([rgb,depth],dim=1).to(self._device)

            with torch.no_grad():
                outputs = self(inputs)
                
            # process images
            img_id = batch_idx*len(valid_it)
            
            for j,x in enumerate(outputs):
                predicted = x
                
                #The squeeze is to remove the channel dimension in preperation
                target = torch.squeeze(target, dim=0)[j].numpy()
                predictions[img_id+j] = torch.squeeze(predicted, dim=0)
                predicted = predicted.argmax(axis=0)


                evaluate.evaluate_image(confusion, results, img_id+j, predictions.numpy(),target)

        # add summary statistics for eval
        results['img'].append('all')
        f1_class_list = []
        for cls_id in confusion:
            f1_measure = evaluate.calculate_f_measure(
                                                    confusion[cls_id]['tp'],
                                                    confusion[cls_id]['fp'],
                                                    confusion[cls_id]['fn'])
            results[f'f_{cls_id}'].append(f1_measure)
            f1_class_list.append((cls_id,f1_measure))
        mcc = evaluate.calculate_mcc(confusion)
        results['mcc'].append(mcc)
        
        
        #results for uncertainity
        auces, auses = evaluate_uncertainty(valid_it, predictions, classes, log_path)
        
        
        #final return
        return f1_class_list, mcc, auces, auses



class MultiViewFusionRGBD(nn.Module):
    def __init__(self,classes, lamda_epochs = 1, device = "cuda"):
        MultiViewFusionRGBD.__init__(self)
        
        #rgb, d u-nets
        self.rgb_unet = UNet(3,classes,"rgb",device)
        self.depth_unet = UNet(1,classes,"d",device)
    
        self.classes = classes
        self.lamda_epochs = lamda_epochs
    
    #https://github.com/Han-Zongbo/TMC/blob/main/TMC%20ICLR/model.py
    def forward(self, rgb_images, depth_images):
        batch_len = rgb_images.shape[0]
        height_len = rgb_images.shape[2]
        width_len = rgb_images.shape[3]
        
        #normal predictions
        rgb_prediction   = self.rgb_unet(rgb_images) #(batch, channel, height, width)
        depth_prediction = self.depth_unet(depth_images)
        
        #dirchlet distribution evidence based on TMC
        rgb_evidence   = ReLU()(rgb_prediction)+1
        depth_evidence = ReLU()(depth_prediction)+1
        
        #combination of evidence and final prediction generation
        alpha_a = torch.zeros(len(self.classes))
        final_prediction = torch.zeros((batch_len,len(self.classes),height_len,width_len)) # (batch, classes, height, width)
        for batch in range(batch_len):
            for height_idx in range(height_len):
                for width_idx in range(width_len):
                    #combination of evidence
                    pixel_evidence = [rgb_evidence[batch,:,height_idx,width_idx], depth_evidence[batch,:,height_idx,width_idx]] # (view=2,classes evidence)
                    alpha_a = self.DS_Combin(pixel_evidence) # (classes_len)
                    final_prediction[batch,:,height_idx,width_idx] = torch.distributions.dirichlet.Dirichlet(alpha_a).sample()
        
        
        return final_prediction
        
        
    def loss_forward(self, rgb_images, depth_images, gt_mask, global_step):
        loss = 0
        batch_len = rgb_images.shape[0]
        height_len = rgb_images.shape[2]
        width_len = rgb_images.shape[3]
        
        #normal predictions
        rgb_prediction   = self.rgb_unet(rgb_images) #(batch, channel, height, width)
        depth_prediction = self.depth_unet(depth_images)
        
        #dirchlet distribution evidence based on TMC
        rgb_evidence   = ReLU()(rgb_prediction)+1
        depth_evidence = ReLU()(depth_prediction)+1
        evidence = [rgb_evidence, depth_evidence] # [(batch, classes, height, width)]
        
        #combination of evidence and final prediction generation
        alpha_a = torch.zeros(len(self.classes))
        final_prediction = torch.zeros((batch_len,len(self.classes),height_len,width_len)) # (batch, classes, height, width)
        for batch in range(batch_len):
            for height_idx in range(height_len):
                for width_idx in range(width_len):
                    #loss calculation
                    temp_loss = 0
                    count = 0
                    pixel_evidence = [rgb_evidence[batch,:,height_idx,width_idx], depth_evidence[batch,:,height_idx,width_idx]] # (view=2,classes evidence)
                    for classifier_evidence in pixel_evidence:
                        temp_loss += self.ce_loss(gt_mask[batch,0,height_idx,width_idx], classifier_evidence, len(self.classes), global_step, self.lamda_epochs)
                        count += 1
                    loss += temp_loss / count
                    
                    #combination of evidence
                    alpha_a = self.DS_Combin(pixel_evidence) # (classes_len)
                    final_prediction[batch,:,height_idx,width_idx] = torch.distributions.dirichlet.Dirichlet(alpha_a).sample()
                    
                    loss += self.ce_loss(gt_mask[batch,0,height_idx,width_idx], alpha_a, len(self.classes), global_step, self.lamda_epochs )
        
        loss = loss / (batch_len*height_len*width_len)
        
        return final_prediction, loss
    
    def _train(self, train_it, optimizer, epoch):
        '''Train model on epoch

        Parameters
        ----------
        train_it : Training data
        criterion : Training criterion
        optimizer : Optimizer
        metric : Metric

        Returns
        -------
        Training loss, metric output

        '''
        self.train()
        epoch_loss = 0.0

        for batch_idx,(rgb, depth, target) in enumerate(train_it):
            target = target.to(self._device)

            rgb_inputs = rgb.to(self._device)
            depth_inputs = depth.to(self._device)

            _, loss = self.loss_forward(rgb_inputs, depth_inputs, target, epoch)

            # add dropout regularization
            reg = torch.zeros(1)  # get the regularization term
            for module in filter(lambda x: isinstance(x, cd.ConcreteDropout2D),
                                 self.modules()):
                reg += module.regularization
            loss += reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss/len(train_it)
        
    def _validate(self, valid_it, epoch):
        '''Validate model

        Parameters
        ----------
        valid_it : Validation data
        criterion : Training criterion
        metric : Metric

        Returns
        -------
        Validation loss, validation metric

        '''
        self.eval()
        epoch_loss = 0.0

        for batch_idx, (rgb, depth, target) in enumerate(valid_it):
            target = target.to(self._device)

            rgb_inputs = rgb.to(self._device)
            depth_inputs = depth.to(self._device)

            with torch.no_grad():
                _, loss = self.loss_forward(rgb_inputs, depth_inputs, target, epoch)

            epoch_loss += loss.item()

        return epoch_loss/len(valid_it)
        
    def fit(self, train_it, valid_it, epochs, log_dir):
        '''Train model

        Parameters
        ----------
        train_it : Training data
        valid_it : Validation data
        epochs : Number of epochs to train
        log_dir : Folder to which logging information is saved
        '''
        
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        checkpoint = ModelCheckpoint(log_dir, self, optimizer)

        epoch_columns = []
        training_losses = []
        validation_losses = []

        for epoch in tqdm(range(epochs)):
            epoch_columns.append(f"Epoch {epoch}")
            
            train_loss = self._train(train_it, optimizer, epoch)
            training_losses.append(train_loss)
            scheduler.step(train_loss)

            valid_loss = self._validate(valid_it, epoch)
            validation_losses.append(valid_loss)
            checkpoint.update(epoch, valid_loss)
        
        pd.DataFrame([epoch_columns,training_losses,validation_losses]).to_csv(os.path.join(log_dir, "Training_and_validation_TMC"), index=False)
        
    def final_evaluation(self, valid_it, classes, log_path):
        self.eval()

        #Prep for eval
        results = {'img': [], 'mcc': []}
        confusion = {f: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for f in classes}
        predictions = torch.zeros((len(valid_it),valid_it.shape[0],valid_it.shape[1],valid_it.shape[2],valid_it.shape[3])) # (Batch,batch_size,channel,height,width)
        for cls_id in classes:
            results[f'f_{cls_id}'] = []


        for batch_idx, (rgb, depth, target) in enumerate(valid_it):
            target = target.to(self._device)

            rgb_inputs = rgb.to(self._device)
            depth_inputs = depth.to(self._device)

            with torch.no_grad():
                outputs = self.forward(rgb_inputs,depth_inputs)
                
            # process images
            img_id = batch_idx*len(valid_it)
            
            for j,x in enumerate(outputs):
                predicted = x
                
                #The squeeze is to remove the channel dimension in preperation
                target = torch.squeeze(target, dim=0)[j].numpy()
                predictions[batch_idx,j] = torch.squeeze(predicted, dim=0)
                predicted = predicted.argmax(axis=0)

                evaluate.evaluate_image(confusion, results, img_id+j, predictions.numpy(),target)

        # add summary statistics for eval
        results['img'].append('all')
        f1_class_list = []
        for cls_id in confusion:
            f1_measure = evaluate.calculate_f_measure(
                                                    confusion[cls_id]['tp'],
                                                    confusion[cls_id]['fp'],
                                                    confusion[cls_id]['fn'])
            results[f'f_{cls_id}'].append(f1_measure)
            f1_class_list.append((cls_id,f1_measure))
        mcc = evaluate.calculate_mcc(confusion)
        results['mcc'].append(mcc)
        
        
        #results for uncertainity
        auces, auses = evaluate_uncertainty(valid_it, predictions, classes, log_path)
        
        
        #final return
        return f1_class_list, mcc, auces, auses
    
    def load(self, model_path):
        '''Load model weights from given path

        Parameters
        ----------
        model_path : Path to the model

        '''
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint['state_dict'])

    # loss function
    def KL(alpha, c):
        beta = torch.ones((1, c)).cuda()
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
        return kl


    def ce_loss(p, alpha, c, global_step, annealing_step):
        '''
        :param p: ground truth
        :param alpha: evidence
        :param c: how many classes there are
        :global_step: which step the epoch is on
        :annealing_step: how many global steps are required for one annealing step
        '''
        S = torch.sum(alpha, dim=1, keepdim=True)
        E = alpha - 1
        label = F.one_hot(p, num_classes=c)
        A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

        annealing_coef = min(1, global_step / annealing_step)

        alp = E * (1 - label) + 1
        B = annealing_coef * self.KL(alp, c)

        return (A + B)

    def DS_Combin(self, alpha):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """
        def DS_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v]-1
                b[v] = E[v]/(S[v].expand(E[v].shape))
                u[v] = self.classes/S[v]

            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, self.classes, 1), b[1].view(-1, 1, self.classes))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))
            # calculate u^a
            u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))

            # calculate new S
            S_a = self.classes / u_a
            # calculate new e_k
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a

        for v in range(len(alpha)-1):
            if v==0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v+1])
        return alpha_a