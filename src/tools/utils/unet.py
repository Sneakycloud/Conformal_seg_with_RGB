import math
import os

import concretedropout.pytorch as cd
import numpy as np
import torch
import evaluate
import pandas as pd
from torch import nn
from tqdm import tqdm
from evaluate_uncertainty_maps import evaluate_uncertainty
from torch.nn import ReLU
import torch.nn.functional as F
from torch.amp import GradScaler


class IoU:

    '''Class-wise intersecion over union metric'''

    def __init__(self, classes, regression=False):
        self._regression = regression
        self._results = {f: [] for f in classes}

    def reset(self):
        for key in self._results:
            self._results[key] = []

    def add(self, predicted, target):
        if self._regression:
            predicted = torch.exp(-torch.exp(predicted))
            target = torch.exp(-torch.exp(target))

        predicted = predicted.argmax(dim=1).view(-1)

        if len(target.shape) == 4:
            target = target.argmax(dim=1).view(-1)
        else:
            target = target.view(-1)

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
                self._results[label].append(intersect / union)

    def value(self):
        results = {}
        for label in self._results:
            vals = self._results[label]

            results[label] = float(np.mean(vals)) if len(vals) > 0 else 0.0
        return results


class WeightedMSE(nn.Module):

    def __init__(self, weights=None, device='cuda'):
        nn.Module.__init__(self)
        if weights is not None:
            self._weights = torch.from_numpy(weights).to(device)
        else:
            self._weights = None

    def forward(self, inputs, targets):
        if len(targets.shape) == 3:
            num_classes = inputs.shape[1]
            targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)
            targets = targets_one_hot.permute(0, 3, 1, 2).float()

        targets = targets.to(inputs.device)

        if self._weights is not None:
            mse = ((inputs - targets) ** 2).mean(dim=(0, 2, 3))
            mse = (self._weights * mse).sum()
        else:
            mse = ((inputs - targets) ** 2).mean()

        return mse


class ModelCheckpoint:

    def __init__(self, log_path, model, optimizer,
                 file_name='minloss_checkpoint.pth.tar'):
        self._model_path = os.path.join(log_path, file_name)
        self._model = model
        self._optimizer = optimizer
        self._minloss = np.inf

    def update(self, epoch, loss):
        if loss < self._minloss:
            self._minloss = loss
            state = {'epoch': epoch, 'state_dict': self._model.state_dict(),
                     'opt_dict': self._optimizer.state_dict()}
            torch.save(state, self._model_path)


class CSVLogger:

    def __init__(self, log_path, metric, file_name='log.csv'):
        self._file = os.path.join(log_path, file_name)
        self._metric_order = [f for f in metric.value()]

        columns = ['epoch', 'loss']
        columns += [f'iou_{f}' for f in self._metric_order]
        columns += ['lr', 'val_loss']
        columns += [f'val_iou_{f}' for f in self._metric_order]

        # --- FIX: append mode so fold logs don't overwrite each other ---
        mode = 'a' if os.path.exists(self._file) else 'w'
        with open(self._file, mode, encoding='utf-8') as log:
            if mode == 'w':
                log.write(','.join(columns) + '\n')

    def update(self, epoch, train_loss, train_metric, valid_loss, valid_metric,
               learning_rate):
        entry = [epoch, train_loss]
        entry += [train_metric[f] for f in self._metric_order]
        entry += [learning_rate, valid_loss]
        entry += [valid_metric[f] for f in self._metric_order]

        with open(self._file, 'a', encoding='utf-8') as log:
            log_entry = ','.join(str(el) for el in entry)
            log.write(log_entry + '\n')


class BaseUNet(nn.Module):

    LAYER_CONFIG = [32, 64, 128, 256]

    @classmethod
    def _pad(cls, inputs):
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
        top, bottom, left, right = padding

        bottom_idx = None if bottom == 0 else -bottom
        right_idx = None if right == 0 else -right

        return inputs[:, :, top:bottom_idx, left:right_idx]

    @classmethod
    def _downsample(cls, inputs, layers, act, pool, steps_per_layer=3):
        result = inputs
        skips = []

        layer_num = len(layers) // steps_per_layer
        for layer_i in range(layer_num):
            conv1 = layers[layer_i*steps_per_layer]
            drop = layers[layer_i*steps_per_layer+1]
            conv2 = layers[layer_i*steps_per_layer+2]

            # --- FIX: removed gradient checkpointing and debug prints ---
            result = drop(result, conv1)
            result = act(result)
            result = conv2(result)
            result = act(result)

            skips.append(result)
            result = pool(result)

        return result, skips

    @classmethod
    def _process(cls, inputs, layers, act):
        conv1 = layers[0]
        drop = layers[1]
        conv2 = layers[2]

        # --- FIX: removed gradient checkpointing and debug prints ---
        result = drop(inputs, conv1)
        result = act(result)
        result = conv2(result)
        result = act(result)

        return result

    @classmethod
    def _upsample(cls, inputs, skips, layers, act, steps_per_layer=4):
        if not inputs.is_cuda:
            skips = [s.cpu() for s in skips]
        result = inputs

        layer_num = len(layers) // steps_per_layer
        for layer_i in range(layer_num):
            tconv = layers[layer_i*steps_per_layer]
            conv1 = layers[layer_i*steps_per_layer+1]
            drop = layers[layer_i*steps_per_layer+2]
            conv2 = layers[layer_i*steps_per_layer+3]

            result = tconv(result)
            result = torch.cat((result, skips.pop()), dim=1)

            # --- FIX: removed gradient checkpointing and debug prints ---
            result = drop(result, conv1)
            result = act(result)
            result = conv2(result)
            result = act(result)

        return result


class UNet(BaseUNet):

    def __init__(self, in_channels, classes, mode, device='cuda'):
        BaseUNet.__init__(self)

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

        for filters in self.LAYER_CONFIG:
            self._downsampling.append(nn.Conv2d(in_channels, filters, 3,
                                      padding='same'))
            in_channels = filters
            self._downsampling.append(cd.ConcreteDropout2D())
            self._downsampling.append(nn.Conv2d(in_channels, filters, 3,
                                      padding='same'))

        self._processing.append(nn.Conv2d(in_channels, 512, 3, padding='same'))
        in_channels = 512
        self._processing.append(cd.ConcreteDropout2D())
        self._processing.append(nn.Conv2d(in_channels, 512, 3, padding='same'))

        for filters in reversed(self.LAYER_CONFIG):
            self._upsampling.append(nn.ConvTranspose2d(in_channels, filters, 2,
                                    stride=2))
            in_channels = filters
            self._upsampling.append(nn.Conv2d(2*in_channels, filters, 3,
                                    padding=1))
            self._upsampling.append(cd.ConcreteDropout2D())
            self._upsampling.append(nn.Conv2d(in_channels, filters, 3,
                                    padding=1))

        self._output = nn.Conv2d(in_channels, len(self._classes), 1)
        self.apply(self._init_weights)
        self.to(device)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Conv2d):
            # --- FIX: use non-deprecated kaiming_normal_ ---
            nn.init.kaiming_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.ConvTranspose2d):
            nn.init.zeros_(layer.bias)

    def forward(self, inputs):
        result, padding = self._pad(inputs)
        result, skips = self._downsample(result, self._downsampling,
                                         self._act, self._pool)
        result = self._process(result, self._processing, self._act)
        result = self._upsample(result, skips, self._upsampling, self._act)
        result = self._output(result)
        return self._unpad(result, padding)

    def load(self, model_path):
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint['state_dict'])

    def fit(self, train_it, valid_it, epochs, log_dir, weights):
        criterion = WeightedMSE(weights)
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        metric = IoU(self._classes)
        checkpoint = ModelCheckpoint(log_dir, self, optimizer)
        csv_log = CSVLogger(log_dir, metric)

        scaler = torch.amp.GradScaler('cuda')

        for epoch in tqdm(range(epochs)):
            train_loss, train_iou = self._train(train_it, criterion,
                                                optimizer, metric, scaler)
            scheduler.step(train_loss)
            valid_loss, valid_iou = self._validate(valid_it, criterion, metric)
            checkpoint.update(epoch, valid_loss)
            csv_log.update(epoch, train_loss, train_iou, valid_loss, valid_iou,
                           optimizer.param_groups[0]['lr'])

    def proba(self, batch):
        self.eval()
        if isinstance(batch, dict):
            batch = batch['inputs'].to(self._device,
                                       dtype=batch['inputs'].dtype)
        predicted = self(batch)
        softmax = torch.nn.Softmax2d()
        predicted = softmax(predicted)
        return predicted.detach().cpu().numpy()

    def predict(self, batch):
        predicted = self.proba(batch)
        return predicted.argmax(axis=1)

    def _set_mc_dropout(self, enable):
        for module in filter(lambda x: isinstance(x, cd.ConcreteDropout2D),
                             self.modules()):
            module.is_mc_dropout = enable

    def mc_dropout(self, batch, samples, batch_size):
        self._set_mc_dropout(True)
        image = batch['inputs'].to(self._device, dtype=batch['inputs'].dtype)
        assert image.shape[0] == 1, 'Process one image at a time'

        batch = image.repeat(batch_size, 1, 1, 1)
        repeats = samples // batch_size
        outputs = []
        for _ in range(repeats):
            predicted = self.proba(batch)
            outputs.append(predicted)

        pred_samples = np.concatenate(outputs)
        pred_distr = pred_samples.mean(axis=0)
        sample_size = pred_samples.shape[0]

        eps = np.finfo(pred_distr.dtype).tiny
        log_distr = np.log(pred_distr + eps)
        pred_entropy = -1 * np.sum(pred_distr * log_distr, axis=0)

        log_samples = np.log(pred_samples + eps)
        minus_e = np.sum(pred_samples * log_samples, axis=(0, 1))
        minus_e /= sample_size
        mutual_info = pred_entropy + minus_e

        self._set_mc_dropout(False)
        return pred_entropy, mutual_info

    def _train(self, train_it, criterion, optimizer, metric, scaler):
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
                inputs = torch.cat([rgb, depth], dim=1).to(self._device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                outputs = self(inputs)

                reg_terms = [m.regularization
                             for m in self.modules()
                             if isinstance(m, cd.ConcreteDropout2D)]
                reg = torch.stack(reg_terms).sum() if reg_terms else torch.tensor(0.0)

                loss = criterion(outputs, target) + reg


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            metric.add(outputs.detach(), target.detach())

        return epoch_loss / len(train_it), metric.value()

    def _validate(self, valid_it, criterion, metric):
        self.eval()
        epoch_loss = 0.0
        metric.reset()

        for (rgb, depth, target) in valid_it:
            target = target.to(self._device)

            if self._mode == "RGB":
                inputs = rgb.to(self._device)
            elif self._mode == "D":
                inputs = depth.to(self._device)
            elif self._mode == "RGBD":
                inputs = torch.cat([rgb, depth], dim=1).to(self._device)

            with torch.no_grad():
                outputs = self(inputs)
                loss = criterion(outputs, target)

            epoch_loss += loss.item()
            metric.add(outputs.detach(), target.detach())

        return epoch_loss / len(valid_it), metric.value()

    def final_evaluation(self, valid_it, classes, log_path):
        self.eval()

        results = {'img': [], 'mcc': []}
        confusion = {f: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for f in classes}

        for cls_id in classes:
            results[f'f_{cls_id}'] = []

        collected_predictions = []

        for batch_idx, (rgb, depth, target) in enumerate(valid_it):
            target = target.to(self._device)

            if self._mode == "RGB":
                inputs = rgb.to(self._device)
            elif self._mode == "D":
                inputs = depth.to(self._device)
            elif self._mode == "RGBD":
                inputs = torch.cat([rgb, depth], dim=1).to(self._device)

            with torch.no_grad():
                outputs = self(inputs)

            img_id = batch_idx * inputs.shape[0]

            for j, x in enumerate(outputs):
                pred_np = x.cpu().numpy()
                collected_predictions.append(pred_np)
                target_np = target[j].squeeze(0).cpu().numpy()
                pred_argmax = pred_np.argmax(axis=0)
                evaluate.evaluate_image(confusion, results, img_id + j,
                                        pred_argmax, target_np)

        results['img'].append('all')
        f1_class_list = []
        for cls_id in confusion:
            f1_measure = evaluate.calculate_f_measure(
                confusion[cls_id]['tp'],
                confusion[cls_id]['fp'],
                confusion[cls_id]['fn'])
            results[f'f_{cls_id}'].append(f1_measure)
            f1_class_list.append((cls_id, f1_measure))
        mcc = evaluate.calculate_mcc(confusion)
        results['mcc'].append(mcc)

        predictions_final = torch.from_numpy(np.stack(collected_predictions))
        auses, auces = evaluate_uncertainty(predictions_final, valid_it,
                                            classes, log_path)

        return f1_class_list, mcc, auces, auses


class MultiViewFusionRGBD(nn.Module):

    def __init__(self, classes, lamda_epochs=1, device='cuda'):
        # --- FIX: was calling MultiViewFusionRGBD.__init__ recursively ---
        super(MultiViewFusionRGBD, self).__init__()

        self.rgb_unet   = UNet(3, classes, "RGB", device)
        self.depth_unet = UNet(1, classes, "D",   device)

        self.classes      = len(classes)
        self.class_list   = classes
        self.lamda_epochs = lamda_epochs
        self._device      = device

    def forward(self, rgb_images, depth_images):

            rgb_prediction   = self.rgb_unet(rgb_images)
            depth_prediction = self.depth_unet(depth_images)

            rgb_evidence   = F.softplus(rgb_prediction)   + 1
            depth_evidence = F.softplus(depth_prediction) + 1

            alpha_combined = self._DS_Combin_vectorised(rgb_evidence, depth_evidence)

            S = alpha_combined.sum(dim=1, keepdim=True)
            probs = alpha_combined / S
            
            if self.training:

                return alpha_combined 
            else:

                return probs

    def _DS_Combin_vectorised(self, alpha1, alpha2):
        '''Vectorised Dempster-Shafer combination of two alpha tensors.

        Parameters
        ----------
        alpha1, alpha2 : (B, C, H, W) tensors of Dirichlet parameters

        Returns
        -------
        alpha_a : (B, C, H, W) combined Dirichlet parameters
        '''
        K = alpha1.shape[1]   # number of classes

        S1 = alpha1.sum(dim=1, keepdim=True)   # (B,1,H,W)
        S2 = alpha2.sum(dim=1, keepdim=True)

        b1 = (alpha1 - 1) / S1   # belief masses  (B,C,H,W)
        b2 = (alpha2 - 1) / S2

        u1 = K / S1              # uncertainty masses  (B,1,H,W)
        u2 = K / S2

        # conflict C = sum_{i!=j} b1_i * b2_j
        # = (sum_i b1_i)(sum_j b2_j) - sum_i b1_i*b2_i
        b1_sum = b1.sum(dim=1, keepdim=True)  # (B,1,H,W)
        b2_sum = b2.sum(dim=1, keepdim=True)
        C = b1_sum * b2_sum - (b1 * b2).sum(dim=1, keepdim=True)  # (B,1,H,W)

        denom = 1.0 - C  # (B,1,H,W)
        denom = denom.clamp(min=1e-8)   # numerical safety

        # combined belief and uncertainty
        b_a = (b1 * b2 + b1 * u2 + b2 * u1) / denom   # (B,C,H,W)
        u_a = (u1 * u2) / denom                        # (B,1,H,W)

        # recover alpha from b_a and u_a
        S_a   = K / u_a.clamp(min=1e-8)
        e_a   = b_a * S_a
        alpha_a = e_a + 1

        return alpha_a   # (B,C,H,W)

    def loss_forward(self, rgb_images, depth_images, gt_mask, global_step):
        '''Compute TMC evidence loss for both views plus combined output'''

        rgb_prediction   = self.rgb_unet(rgb_images)
        depth_prediction = self.depth_unet(depth_images)

        rgb_evidence   = F.softplus(rgb_prediction)   + 1
        depth_evidence = F.softplus(depth_prediction) + 1

        # per-view loss
        loss_rgb   = self._edl_loss(rgb_evidence,   gt_mask, global_step)
        loss_depth = self._edl_loss(depth_evidence, gt_mask, global_step)

        # combined loss
        alpha_combined = self._DS_Combin_vectorised(rgb_evidence, depth_evidence)
        loss_combined  = self._edl_loss(alpha_combined, gt_mask, global_step)

        return alpha_combined, (loss_rgb + loss_depth + loss_combined) / 3.0

    def _edl_loss(self, alpha, gt_mask, global_step):
        '''Vectorised EDL loss over all pixels.

        Parameters
        ----------
        alpha   : (B, C, H, W) Dirichlet parameters
        gt_mask : (B, H, W) or (B, 1, H, W) integer class labels
        '''
        if gt_mask.dim() == 4:
            gt_mask = gt_mask.squeeze(1)   # (B, H, W)

        B, C, H, W = alpha.shape

        # flatten spatial dims for batched ops
        alpha_flat = alpha.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        labels_flat = gt_mask.reshape(-1).long()                # (B*H*W,)

        S = alpha_flat.sum(dim=1, keepdim=True)                 # (B*H*W, 1)
        one_hot = F.one_hot(labels_flat, num_classes=C).float() # (B*H*W, C)

        # classification
        A = (one_hot * (torch.digamma(S) - torch.digamma(alpha_flat))).sum(dim=1)

        # KL regularisation with annealing
        annealing_coef = min(1.0, global_step / max(int(self.lamda_epochs), 1))
        alpha_tilde = (alpha_flat-1) * (1-one_hot) + 1
        loss_kl = self._KL_flat(alpha_tilde, C)

        return (A + annealing_coef * loss_kl.squeeze(1)).mean()

    def _KL_flat(self, alpha, C):
        '''KL divergence between Dir(alpha) and Dir(1,...,1).  (N, C) -> (N, 1)'''
        beta  = torch.ones_like(alpha)
        S_a   = alpha.sum(dim=1, keepdim=True)
        S_b   = torch.tensor(float(C), device=alpha.device)

        lnB      = torch.lgamma(S_a) - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        lnB_uni  = torch.lgamma(beta).sum(dim=1, keepdim=True) - torch.lgamma(S_b)
        dg_diff  = torch.digamma(alpha) - torch.digamma(S_a)

        kl = ((alpha - beta) * dg_diff).sum(dim=1, keepdim=True) + lnB + lnB_uni
        return kl

    def fit(self, train_it, valid_it, epochs, log_dir, weights=None):
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        checkpoint = ModelCheckpoint(log_dir, self, optimizer,
                                     file_name='minloss_checkpoint_TMC.pth.tar')

        training_losses   = []
        validation_losses = []

        scaler = torch.amp.GradScaler('cuda')

        for epoch in tqdm(range(epochs)):
            train_loss = self._train(train_it, optimizer, epoch, scaler)
            training_losses.append(train_loss)
            scheduler.step(train_loss)

            valid_loss = self._validate(valid_it, epoch)
            validation_losses.append(valid_loss)
            checkpoint.update(epoch, valid_loss)
            print(f'  [TMC] epoch {epoch}  train={train_loss:.4f}  val={valid_loss:.4f}',
                  flush=True)

        pd.DataFrame({
            'epoch': list(range(epochs)),
            'train_loss': training_losses,
            'val_loss': validation_losses
        }).to_csv(os.path.join(log_dir, 'Training_and_validation_TMC.csv'), index=False)

    def _train(self, train_it, optimizer, epoch, scaler):
        self.train()
        epoch_loss = 0.0

        for (rgb, depth, target) in train_it:
            target      = target.to(self._device)
            rgb_inputs  = rgb.to(self._device)
            depth_inputs = depth.to(self._device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                _, loss = self.loss_forward(rgb_inputs, depth_inputs,
                                            target, epoch)

                # concrete dropout regularisation from both sub-networks
                reg_terms = [m.regularization
                             for m in self.modules()
                             if isinstance(m, cd.ConcreteDropout2D)]
                reg = torch.stack(reg_terms).sum() if reg_terms else torch.tensor(0.0)
                loss = loss + reg

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        return epoch_loss / len(train_it)

    def _validate(self, valid_it, epoch):
        self.eval()
        epoch_loss = 0.0

        for (rgb, depth, target) in valid_it:
            target       = target.to(self._device)
            rgb_inputs   = rgb.to(self._device)
            depth_inputs = depth.to(self._device)

            with torch.no_grad():
                _, loss = self.loss_forward(rgb_inputs, depth_inputs,
                                            target, epoch)

            epoch_loss += loss.item()

        return epoch_loss / len(valid_it)

    def final_evaluation(self, valid_it, classes, log_path):
        self.eval()

        results = {'img': [], 'mcc': []}
        confusion = {f: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for f in classes}

        for cls_id in classes:
            results[f'f_{cls_id}'] = []

        collected_predictions = []

        for batch_idx, (rgb, depth, target) in enumerate(valid_it):
            target       = target.to(self._device)
            rgb_inputs   = rgb.to(self._device)
            depth_inputs = depth.to(self._device)

            with torch.no_grad():

                outputs = self.forward(rgb_inputs, depth_inputs)

            img_id = batch_idx * rgb.shape[0]

            for j, x in enumerate(outputs):
                pred_np   = x.cpu().numpy()
                collected_predictions.append(pred_np)
                target_np = target[j].squeeze(0).cpu().numpy()
                pred_argmax = pred_np.argmax(axis=0)
                evaluate.evaluate_image(confusion, results, img_id + j,
                                        pred_argmax, target_np)

        results['img'].append('all')
        f1_class_list = []
        for cls_id in confusion:
            f1_measure = evaluate.calculate_f_measure(
                confusion[cls_id]['tp'],
                confusion[cls_id]['fp'],
                confusion[cls_id]['fn'])
            results[f'f_{cls_id}'].append(f1_measure)
            f1_class_list.append((cls_id, f1_measure))
        mcc = evaluate.calculate_mcc(confusion)
        results['mcc'].append(mcc)

        predictions_final = torch.from_numpy(np.stack(collected_predictions))
        auses, auces = evaluate_uncertainty(predictions_final, valid_it,
                                            classes, log_path)

        return f1_class_list, mcc, auces, auses

    def load(self, model_path):
        ckpt = torch.load(model_path)
        self.load_state_dict(ckpt['state_dict'])

    def KL(self, alpha, c):
        beta    = torch.ones((1, c)).to(alpha.device)
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta  = torch.sum(beta,  dim=1, keepdim=True)
        lnB     = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
        dg0     = torch.digamma(S_alpha)
        dg1     = torch.digamma(alpha)
        kl      = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
        return kl

    def ce_loss(self, p, alpha, c, global_step, annealing_step):
    
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
        A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)),
                          dim=1, keepdim=True)
        annealing_coef = min(1, global_step / annealing_step)
        alp = E * (1 - label) + 1
        B = annealing_coef * self.KL(alp, c)
        return A + B
