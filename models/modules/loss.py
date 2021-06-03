import torch
import torchvision
from torch import nn as nn

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
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
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None



class ContrastiveLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, nb_preds=1):
        super().__init__()
        self.nb_preds = nb_preds
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def __call__(self, pred_true,pred_false):
        current_batch_size=pred_true.shape[0]
        temp=torch.cat((pred_true.flatten().unsqueeze(1),pred_false.flatten().unsqueeze(0).repeat(self.nb_preds*current_batch_size,1)),dim=1)
        loss= self.cross_entropy_loss(temp,torch.zeros(temp.shape[0], dtype=torch.long,device=temp.device))
        
        return loss.mean()


class DiscriminatorLoss(nn.Module):
    def __init__(self,opt,netD,device):
        super().__init__()
        self.opt=opt
        self.device=device
    
    def compute_loss_D(self,netD,real,fake):
        pass

    def compute_loss_G(self,netD,real,fake):
        pass

class DiscriminatorGANLoss(DiscriminatorLoss):
    def __init__(self,opt,netD,device):
        super().__init__(opt,netD,device)
        if opt.D_label_smooth:
            target_real_label = 0.9
        else:
            target_real_label = 1.0
        self.criterionGAN = GANLoss(opt.gan_mode,target_real_label=target_real_label).to(self.device)
        
    def compute_loss_D(self,netD,real,fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        lambda_loss=0.5
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * lambda_loss
        return loss_D

    def compute_loss_G(self,netD,real,fake):
        pred_fake = netD(fake)
        loss_D_fake = self.criterionGAN(pred_fake,True)
        return loss_D_fake
        
class DiscriminatorContrastiveLoss(DiscriminatorLoss):
    def __init__(self,opt,netD,device):
        super().__init__(opt,netD,device)
        self.nb_preds=int(torch.prod(torch.tensor(netD(torch.zeros([1,opt.input_nc,opt.crop_size,opt.crop_size], dtype=torch.float,device=self.device)).shape)))
        self.criterionContrastive = ContrastiveLoss(self.nb_preds)
        
    def compute_loss_D(self,netD,real,fake):
        """Calculate contrastive GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake = fake.detach()
        pred_fake = netD(fake)
        # Real
        pred_real = netD(real)
        
        loss_D_real = self.criterionContrastive(pred_real,pred_fake)
        loss_D_fake = self.criterionContrastive(-pred_fake,-pred_real)
        
        # combine loss and calculate gradients
        return (loss_D_fake + loss_D_real) * 0.5

    def compute_loss_G(self,netD,real,fake):
        loss_G = self.criterionContrastive(-netD(real),-netD(fake))
        return loss_G
