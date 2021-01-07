from settings import warmup_epochs, decay_start, epochs_to_zero_lr


def lambda_rule(epoch):
    lr_l = 1.0 - max(0, epoch + warmup_epochs - decay_start) / float(epochs_to_zero_lr + 1)
    return lr_l
