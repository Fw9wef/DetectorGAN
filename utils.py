from settings import warmup_epochs, decay_start, epochs_to_zero_lr, gan_pretrain_epochs
import torch


def gan_lambda_rule(epoch):
    lr_l = 1.0 - max(0, epoch + warmup_epochs - decay_start) / float(epochs_to_zero_lr + 1)
    return lr_l


def detector_lambda_rule(epoch):
    lr_l = 1.0 - max(0, epoch + warmup_epochs + gan_pretrain_epochs - decay_start) / float(epochs_to_zero_lr + 1)
    return lr_l


def centers2vertices(centers):
    vertices = torch.zeros_like(centers)
    vertices[:, 0], vertices[:, 2] = centers[:, 0] - centers[:, 2] / 2, centers[:,0] + centers[:, 2] / 2
    vertices[:, 1], vertices[:, 3] = centers[:, 1] - centers[:, 3] / 2, centers[:, 1] + centers[:, 3] / 2
    return vertices


def find_intersection(set_1, set_2):
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]


def iou(set_1, set_2):
    set_1, set_2 = centers2vertices(set_1), centers2vertices(set_2)

    intersection = find_intersection(set_1, set_2)

    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])

    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection

    return intersection / union  # (n1, n2)
