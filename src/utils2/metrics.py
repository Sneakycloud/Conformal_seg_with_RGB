import torch

def compute_iou(pred, target, num_classes):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            continue  # Skips class if it's missing entirely from validation frame
        ious.append(intersection / union)

    return torch.mean(torch.tensor(ious)) if ious else torch.tensor(0.0)