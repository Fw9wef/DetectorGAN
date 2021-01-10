import functools
from utils import iou
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from image_pool import ImagePool
import settings


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        class_name = m.__class__.__name__
        if hasattr(m, 'weight') and (class_name.find('Conv') != -1 or class_name.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif class_name.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9):
        super(Generator, self).__init__()
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.input_nc = input_nc
        self.output_nc = output_nc

        self.conv1 = nn.Conv2d(input_nc, ngf, 7, 1, 0)
        self.conv1_norm = nn.InstanceNorm2d(ngf)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 3, 2, 1)
        self.conv2_norm = nn.InstanceNorm2d(ngf * 2)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1)
        self.conv3_norm = nn.InstanceNorm2d(ngf * 4)

        self.resnet_blocks = []
        for i in range(n_blocks):
            self.resnet_blocks.append(resnet_block(ngf * 4, 3, 1, 1))
            self.resnet_blocks[i].weight_init(0, 0.02)

        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        self.deconv1 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1)
        self.deconv1_norm = nn.InstanceNorm2d(ngf * 2)
        self.deconv2 = nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1)
        self.deconv2_norm = nn.InstanceNorm2d(ngf)
        self.deconv3 = nn.Conv2d(ngf, self.output_nc, 7, 1, 0)

        self.tanh = torch.nn.Tanh()

    def forward(self, input_):
        x = F.pad(input_, (3, 3, 3, 3), 'reflect')
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.relu(self.conv2_norm(self.conv2(x)))
        x = F.relu(self.conv3_norm(self.conv3(x)))
        x = self.resnet_blocks(x)
        x = F.relu(self.deconv1_norm(self.deconv1(x)))
        x = F.relu(self.deconv2_norm(self.deconv2(x)))
        x = F.pad(x, (3, 3, 3, 3), 'reflect')
        x = self.deconv3(x)
        image = self.tanh(x)
        return image


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class resnet_block(nn.Module):
    def __init__(self, channel, kernel, stride, padding):
        super(resnet_block, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.conv1 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv2_norm = nn.InstanceNorm2d(channel)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input_):
        x = F.pad(input_, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = self.conv2_norm(self.conv2(x))

        return input_ + x


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input_):
        return self.model(input_)


class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        loss = 0
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class CropDrones(nn.Module):
    def __init__(self):
        super(CropDrones, self).__init__()
        # self.register_buffer('max_side', torch.tensor(max_side))

    def forward(self, input_):
        input1, input2 = input_
        images = input1[:, :-1]
        windows = input1[:, -1]
        crops = torch.zeros_like(input2)

        max_side = crops.shape[-1]

        for i, window, image in zip(range(windows.shape[0]), windows, images):
            inds = torch.nonzero(window, as_tuple=False)
            tops = torch.max(inds, dim=0)[0]
            bots = torch.min(inds, dim=0)[0]
            sides = tops - bots
            indent = max_side - sides

            left_indent = indent[1] // 2
            right_indent = indent[1] // 2 + indent[1] % 2
            top_indent = indent[0] // 2
            bot_indent = indent[0] // 2 + indent[0] % 2

            crops[i, :, top_indent:-bot_indent, left_indent:-right_indent] = image[:, bots[0]:tops[0], bots[1]:tops[1]]

        return crops


class GAN(nn.Module):
    def __init__(self, lambda_ABA=settings.lambda_ABA, lambda_BAB=settings.lambda_BAB,
                 lambda_local=settings.lambda_local, pool_size=settings.pool_size, max_crop_side=settings.max_crop_side,
                 decay_start=settings.decay_start, epochs_to_zero_lr=settings.epochs_to_zero_lr,
                 warm_epochs=settings.warmup_epochs):
        super(GAN, self).__init__()

        self.r = 0
        self.lambda_ABA = lambda_ABA
        self.lambda_BAB = lambda_BAB
        self.lambda_local = lambda_local
        self.max_crop_side = max_crop_side

        self.netG_A = Generator(input_nc=4, output_nc=3)
        self.netG_B = Generator(input_nc=4, output_nc=3)
        self.netD_A = NLayerDiscriminator(input_nc=3)
        self.netD_B = NLayerDiscriminator(input_nc=3)
        self.localD = NLayerDiscriminator(input_nc=3)
        self.crop_drones = CropDrones()
        self.criterionGAN = GANLoss("lsgan")
        self.criterionCycle = nn.L1Loss()

        init_weights(self.netG_A)
        init_weights(self.netG_B)
        init_weights(self.netD_A)
        init_weights(self.netD_B)
        init_weights(self.localD)

        self.fake_B_pool = ImagePool(pool_size)
        self.fake_A_pool = ImagePool(pool_size)
        self.fake_drones_pool = ImagePool(pool_size)



    def get_inputs(self, input_):
        self.real_A_with_windows = torch.as_tensor(input_['A'], device=self.device)
        self.real_B_with_windows = torch.as_tensor(input_['B'], device=self.device)
        self.real_A = self.real_A_with_windows[:, :-1]
        self.real_B = self.real_B_with_windows[:, :-1]
        self.A_windows = self.real_A_with_windows[:, -1:]
        self.B_windows = self.real_B_with_windows[:, -1:]
        self.real_drones = torch.zeros(self.real_B.shape[0], 3, self.max_crop_side, self.max_crop_side,
                                       device=self.device)
        self.fake_drones = torch.zeros(self.real_A.shape[0], 3, self.max_crop_side, self.max_crop_side,
                                       device=self.device)

    def forward(self, input_):
        self.get_inputs(input_)
        self.fake_A = self.netG_A(self.real_B_with_windows)
        self.rest_B = self.netG_B(torch.cat([self.fake_A, self.B_windows], dim=1))
        self.real_drones = self.crop_drones((self.real_B_with_windows, self.real_drones))

        self.fake_B = self.netG_B(self.real_A_with_windows)
        self.rest_A = self.netG_A(torch.cat([self.fake_B, self.A_windows], dim=1))
        self.fake_drones = self.crop_drones((torch.cat([self.fake_B, self.A_windows], dim=1), self.fake_drones))

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def iteration(self, input_):
        self.forward(input_)
        loss_dict = dict()

        # backward for D_A
        real_output_D_A = self.netD_A(self.real_A)
        real_GAN_loss_D_A = self.criterionGAN(real_output_D_A, True)

        fake_A = self.fake_B_pool.query(self.fake_A)
        fake_output_D_A = self.netD_A(fake_A.detach())
        fake_GAN_loss_D_A = self.criterionGAN(fake_output_D_A, False)

        D_A_loss = (real_GAN_loss_D_A + fake_GAN_loss_D_A) * 0.5
        loss_dict['D_A'] = D_A_loss

        # backward for D_B
        real_output_D_B = self.netD_B(self.real_B)
        real_GAN_loss_D_B = self.criterionGAN(real_output_D_B, True)

        fake_B = self.fake_B_pool.query(self.fake_B)
        fake_output_D_B = self.netD_B(fake_B.detach())
        fake_GAN_loss_D_B = self.criterionGAN(fake_output_D_B, False)

        D_B_loss = (real_GAN_loss_D_B + fake_GAN_loss_D_B) * 0.5
        loss_dict['D_B'] = D_B_loss

        # backward for localD
        real_output_localD = self.localD(self.real_drones)
        real_GAN_loss_localD = self.criterionGAN(real_output_localD, True)

        fake_drones = self.fake_drones_pool.query(self.fake_drones)
        fake_output_localD = self.localD(fake_drones.detach())
        fake_GAN_loss_localD = self.criterionGAN(fake_output_localD, False)

        localD_loss = (real_GAN_loss_localD + fake_GAN_loss_localD) * 0.5
        loss_dict['local_D'] = localD_loss

        # backward for G_A and G_B
        G_A_GAN_loss = self.criterionGAN(self.netD_A(self.fake_A), True)
        BAB_cycle_loss = self.criterionCycle(self.real_B, self.rest_B)

        G_B_GAN_loss = self.criterionGAN(self.netD_B(self.fake_B), True)
        G_B_local_loss = self.criterionGAN(self.localD(self.fake_drones), True)
        ABA_cycle_loss = self.criterionCycle(self.real_A, self.rest_A)

        G_loss = G_B_GAN_loss + G_A_GAN_loss + G_B_local_loss * self.lambda_local + ABA_cycle_loss *\
                 self.lambda_ABA * self.r + BAB_cycle_loss * self.lambda_BAB * self.r

        loss_dict['G_B'] = G_B_GAN_loss
        loss_dict['G_A'] = G_A_GAN_loss
        loss_dict['G_local'] = G_B_local_loss
        loss_dict['G'] = G_loss
        return loss_dict


class DetectionLayer(nn.Module):
    def __init__(self, n_layers, n_classes=settings.n_classes):
        super(DetectionLayer, self).__init__()
        self.classifier = nn.Sequential(nn.Conv2d(n_classes+1, n_layers, kernel_size=1), nn.Softmax(dim=1))
        self.detector = nn.Sequential(nn.Conv2d(4, n_layers, kernel_size=1), nn.Tanh())

    def forward(self, input_):
        return {'clf': self.classifier(input_), 'reg': self.detector(input_)}


class DetectorCriterion(nn.Module):
    def __init__(self, priors, negative_ratio=settings.negative_ratio, n_classes=settings.n_classes, reg_alpha = settings.reg_alpha):
        super(DetectorCriterion, self).__init__()
        self.priors = torch.cat([prior.reshape(-1, 4) for prior in priors], dim=0)
        self.negative_ratio = negative_ratio
        self.n_classes = n_classes + 1
        self.reg_alpha = reg_alpha
        self.clf_criterion = nn.NLLLoss(reduction='none')
        self.reg_criterion = nn.MSELoss(reduction='mean')

    def forward(self, input_, target_):
        clf_loss = reg_loss = 0

        clf_preds = torch.cat([x['clf'].permute(0, 2, 3, 1).reshape(target_.shape[0], -1, self.n_classes) for x in input_], dim=0)
        reg_preds = torch.cat([x['reg'].permute(0, 2, 3, 1).reshape(target_.shape[0], -1, 4) for x in input_], dim=0)

        for img_n, (objects_, clf_, reg_) in enumerate(zip(target_, clf_preds, reg_preds)):
            prior_objects_iou = iou(self.priors, objects_) # (n_objects, n_prior_boxies)
            _, best_iou_inds = prior_objects_iou.max(dim=1)
            prior_objects_iou[best_iou_inds] = 1.
            _, objects_for_each_prior = prior_objects_iou.max(dim=0)
            targets_for_each_prior
            positives = targets_for_each_prior > 0
            negatives = torch.logical_not(positives)
            n_positives = positives.sum()
            n_negatives = self.negative_ratio * n_positives

            # calculating positives loss for image
            pos_clf_preds, pos_reg_preds = clf_[positives], reg_[positives]
            pos_clf_targets = targets_for_each_prior[positives]
            clf_loss += torch.mean(self.clf_criterion(pos_clf_preds, pos_clf_targets.long()))
            reg_loss += self.reg_criterion()    ###############

            # calculating negatives loss for image
            neg_clf_preds, neg_reg_preds = clf_[negatives], reg_[negatives]
            neg_clf_targets = targets_for_each_prior[negatives]
            neg_clf_loss, _ = self.clf_criterion().sort(descending=True)
            clf_loss += neg_clf_loss[:n_negatives]

        clf_loss /= target_.shape[0]
        reg_loss /= target_.shape[0]
        loss = clf_loss + self.reg_alpha * reg_loss
        return loss


class Detector(nn.Module):
    def __init__(self, load_pretrained_detector = settings.load_pretrained_detector):
        super(Detector, self).__init__()
        from torchvision.models import resnet34
        resnet_conv_layers = list(resnet34(pretrained = load_pretrained_detector).children())[-2]

        # define backbone
        self.backbone = nn.Sequential(*resnet_conv_layers[:-1])

        test_input = torch.rand((1, 3, 256, 256))
        test_input = self.backbone(test_input)

        # define detection layers
        self.priors = self.get_priors(16, [0.2]) + self.get_priors(8, [0.4])
        self.detector16 = []
        for _ in self.priors16:
            self.detector16.append(DetectionLayer(test_input.shape[1]))

        self.downsample = resnet_conv_layers[-1]
        test_input = self.downsample(test_input)
        self.detector8 = []
        for _ in self.priors8:
            self.detector8.append(DetectionLayer(test_input.shape[1]))

        # define detector criterions
        self.criterion = DetectorCriterion(self.priors)

    def get_priors(self, side_div, anchor_side, aspect_ratios=settings.aspect_ratios, side_res=256):
        prior_tensors = []
        xy = torch.arange(0,1,1/side_div)+1/side_div/2
        yy, xx = torch.meshgrid(xy,xy)
        for a in anchor_side:
            for aspect in aspect_ratios:
                w, h = torch.ones_like(yy)*a*np.sqrt(aspect), torch.ones_like(yy)*a/np.sqrt(aspect)
                prior_tensor = torch.stack([yy, xx, h, w], dim=-1)
                prior_tensors.append(prior_tensor)
        return prior_tensors

    def forward(self, input_):
        features = self.backbone(input_)

        detection16 = []
        for detector in self.detector16:
            detection16.append(detector(features))

        features = self.downample(features)

        detection8 = []
        for detector in self.detector8:
            detection8.append(detector(features))

        return detection16 + detection8

    def iteration(self, input_, target_):
        detection = self.forward(input_)
        loss = self.criterion(detection, target_)
        return loss

    def detect_objects(self, input_):
        detection = self.forward(input_)
        suppressed_boxies = self.nms(detection)
        return suppressed_boxies

    def nms(self, detections):
        return [0]
