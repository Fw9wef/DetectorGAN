import torch
from torch.optim import lr_scheduler
from models import GAN, Detector
import itertools
from utils import *


GAN_model = GAN()
Detector_model = Detector()

beta1 = 0.5
lr = 0.0002
optimizer_G = torch.optim.Adam(itertools.chain(GAN_model.netG_A.parameters(),
                                               GAN_model.netG_B.parameters()), lr=lr, betas=(beta1, 0.999))
optimizer_D = torch.optim.Adam(itertools.chain(GAN_model.netD_A.parameters(),
                                               GAN_model.netD_B.parameters(),
                                               GAN_model.localD.parameters()), lr=lr, betas=(beta1, 0.999))


schedulers = list()
schedulers.append(lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule))
schedulers.append(lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_rule))
