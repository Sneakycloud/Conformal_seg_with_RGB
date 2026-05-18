import torch
import numpy as np

def calibrate_fcp(model, calibration_loader, significance_level=0.1):
    """ Computes the conformal threshold (q_hat) using a calibration set. """
    model.eval()
    non_conformity_scores = []

    with torch.no_grad():
        for images, masks in calibration_loader:
            images, masks = images.cuda(), masks.cuda()
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            
            # Non-conformity score calculation: 1 - prob_of_true_class
            true_class_probs = torch.gather(probs, 1, masks.unsqueeze(1)).squeeze(1)
            scores = 1.0 - true_class_probs
            non_conformity_scores.append(scores.cpu().numpy().flatten())

    non_conformity_scores = np.concatenate(non_conformity_scores)
    n = len(non_conformity_scores)
    q_level = np.clip((n + 1) * (1 - significance_level) / n, 0, 1)
    q_hat = np.quantile(non_conformity_scores, q_level)
    return q_hat

def predict_fcp(model, image, q_hat):
    """ Returns a prediction set (tensor of booleans) for each pixel. """
    model.eval()
    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)
        prediction_sets = (1.0 - probs) <= q_hat
    return prediction_sets