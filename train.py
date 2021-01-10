import settings
import torch
from torch.optim import lr_scheduler
from models import GAN, Detector
from data import get_dataloader
import itertools
from utils import *


GAN_model = GAN()
Detector_model = Detector()
dataset = get_dataloader(batch_size=settings.batch_size, num_workers=settings.num_workers)

beta1 = 0.5
gan_lr = 0.0002
detector_lr = 1e-3

optimizer_G = torch.optim.Adam(itertools.chain(GAN_model.netG_A.parameters(),
                                               GAN_model.netG_B.parameters()),
                               lr=gan_lr, betas=(beta1, 0.999))
optimizer_D = torch.optim.Adam(itertools.chain(GAN_model.netD_A.parameters(),
                                               GAN_model.netD_B.parameters(),
                                               GAN_model.localD.parameters()),
                               lr=gan_lr, betas=(beta1, 0.999))
optimizer_Detector = torch.optim.Adam(itertools.chain(Detector_model.backbone,
                                                      Detector_model.detector16,
                                                      Detector_model.detector8),
                                      lr=detector_lr)

schedulers = list()
schedulers.append(lr_scheduler.LambdaLR(optimizer_G, lr_lambda=gan_lambda_rule))
schedulers.append(lr_scheduler.LambdaLR(optimizer_D, lr_lambda=gan_lambda_rule))
schedulers.append(lr_scheduler.LambdaLR(optimizer_Detector, lr_lambda=detector_lambda_rule))


for epoch in range(1, settings.gan_pretrain_epochs + 1):
    for batch_n, batch in enumerate(dataset):
        pass

for epoch in range(1, settings.joint_train_epochs + 1):
    for batch_n, batch in enumerate(dataset):
        pass
