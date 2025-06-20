import functools

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.utils import spectral_norm
from torch.nn import functional as F
###############################################################################
# Functions
###############################################################################


def weights_init_normal(m):
  """Initializes the weights of a module with a normal distribution.

  Args:
    m: The module to initialize.
  """
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Sequential):
        return
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
  """Initializes the weights of a module with a Xavier normal distribution.

  Args:
    m: The module to initialize.
  """
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
  """Initializes the weights of a module with a Kaiming normal distribution.

  Args:
    m: The module to initialize.
  """
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
  """Initializes the weights of a module with an orthogonal distribution.

  Args:
    m: The module to initialize.
  """
    classname = m.__class__.__name__
    print(classname)
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        init.orthogonal(m.weight.data, gain=1)
    elif isinstance(m, nn.Linear):
        init.orthogonal(m.weight.data, gain=1)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
  """Initializes the weights of a network.

  Args:
    net: The network to initialize.
    init_type: The type of initialization to use (e.g., 'normal', 'xavier', 'kaiming', 'orthogonal').
  """
    print('[i] initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    elif init_type == 'edsr':
        pass
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
  """Returns a normalization layer.

  Args:
    norm_type: The type of normalization layer to use (e.g., 'batch', 'instance', 'none').

  Returns:
    A normalization layer.
  """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_D(opt, in_channels=3):
  """Defines a discriminator network.

  Args:
    opt: The options for the discriminator network.
    in_channels: The number of input channels.

  Returns:
    A discriminator network.
  """
    # use_sigmoid = opt.gan_type == 'gan'
    use_sigmoid = False # incorporate sigmoid into BCE_stable loss

    if opt.which_model_D == 'disc_vgg':
        netD = Discriminator_VGG(in_channels, use_sigmoid=use_sigmoid)
        init_weights(netD, init_type='kaiming')
    elif opt.which_model_D == 'disc_patch':
        netD = NLayerDiscriminator(in_channels, 64, 3, nn.InstanceNorm2d, use_sigmoid, getIntermFeat=False)
        init_weights(netD, init_type='normal')
    elif opt.which_model_D == 'disc_unet':
        netD = UNetDiscriminatorSN(in_channels) 
    else:
        raise NotImplementedError('%s is not implemented' %opt.which_model_D)

    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(opt.gpu_ids[0])
    
    return netD


def print_network(net):
  """Prints the network architecture and the number of parameters.

  Args:
    net: The network to print.
  """
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
    print('The size of receptive field: %d' % receptive_field(net))


def receptive_field(net):
  """Computes the receptive field of a network.

  Args:
    net: The network to compute the receptive field for.

  Returns:
    The size of the receptive field.
  """
    def _f(output_size, ksize, stride, dilation):
        return (output_size - 1) * stride + ksize * dilation - dilation + 1

    stats = []
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            stats.append((m.kernel_size, m.stride, m.dilation))
    
    rsize = 1
    for (ksize, stride, dilation) in reversed(stats):
        if type(ksize) == tuple: ksize = ksize[0]
        if type(stride) == tuple: stride = stride[0]
        if type(dilation) == tuple: dilation = dilation[0]
        rsize = _f(rsize, ksize, stride, dilation)
    return rsize


def debug_network(net):
  """Registers a forward hook to print the output size of each module in a network.

  Args:
    net: The network to debug.
  """
    def _hook(m, i, o):
        print(o.size())
    for m in net.modules():
        m.register_forward_hook(_hook)


##############################################################################
# Classes
##############################################################################

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
  """Defines a PatchGAN discriminator.

  Args:
    input_nc: The number of input channels.
    ndf: The number of discriminator filters in the first convolutional layer.
    n_layers: The number of convolutional layers in the discriminator.
    norm_layer: The normalization layer to use.
    use_sigmoid: Whether to use a sigmoid activation function at the end.
    branch: The number of branches in the discriminator.
    bias: Whether to use bias in convolutional layers.
    getIntermFeat: Whether to return intermediate features.
  """
    def __init__(self, input_nc, ndf=64, n_layers=3,
    norm_layer=nn.BatchNorm2d, use_sigmoid=False,
    branch=1, bias=True, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc*branch, ndf*branch, kernel_size=kw, stride=2, padding=padw, groups=branch, bias=True), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev*branch, nf*branch, groups=branch, kernel_size=kw, stride=2, padding=padw, bias=bias),
                norm_layer(nf*branch), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev*branch, nf*branch, groups=branch, kernel_size=kw, stride=1, padding=padw, bias=bias),
            norm_layer(nf*branch),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf*branch, 1*branch, groups=branch, kernel_size=kw, stride=1, padding=padw, bias=True)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
      """Performs a forward pass through the NLayerDiscriminator.

      Args:
        input: The input tensor.

      Returns:
        The output of the discriminator. If getIntermFeat is True, returns a list of intermediate features.
      """
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


class Discriminator_VGG(nn.Module):
  """Defines a VGG-style discriminator.

  Args:
    in_channels: The number of input channels.
    use_sigmoid: Whether to use a sigmoid activation function at the end.
  """
    def __init__(self, in_channels=3, use_sigmoid=True):
        super(Discriminator_VGG, self).__init__()
        def conv(*args, **kwargs):
            return nn.Conv2d(*args, **kwargs)

        num_groups = 32

        body = [
            conv(in_channels, 64, kernel_size=3, padding=1), # 224
            nn.LeakyReLU(0.2),

            conv(64, 64, kernel_size=3, stride=2, padding=1), # 112
            nn.GroupNorm(num_groups, 64),
            nn.LeakyReLU(0.2),

            conv(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, 128),
            nn.LeakyReLU(0.2),

            conv(128, 128, kernel_size=3, stride=2, padding=1), # 56
            nn.GroupNorm(num_groups, 128),
            nn.LeakyReLU(0.2),

            conv(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, 256),
            nn.LeakyReLU(0.2),

            conv(256, 256, kernel_size=3, stride=2, padding=1), # 28
            nn.GroupNorm(num_groups, 256),
            nn.LeakyReLU(0.2),

            conv(256, 512, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, 512),
            nn.LeakyReLU(0.2),

            conv(512, 512, kernel_size=3, stride=2, padding=1), # 14
            nn.GroupNorm(num_groups, 512),
            nn.LeakyReLU(0.2),

            conv(512, 512, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups, 512),
            nn.LeakyReLU(0.2),

            conv(512, 512, kernel_size=3, stride=2, padding=1), # 7
            nn.GroupNorm(num_groups, 512),
            nn.LeakyReLU(0.2),
        ]

        tail = [
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        ]

        if use_sigmoid:
            tail.append(nn.Sigmoid())
        
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
      """Performs a forward pass through the Discriminator_VGG.

      Args:
        x: The input tensor.

      Returns:
        The output of the discriminator.
      """
        x = self.body(x)
        out = self.tail(x)
        return out

class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN).

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x, illu = None):
      """Performs a forward pass through the UNetDiscriminatorSN.

      Args:
        x: The input tensor.
        illu: The illumination map. (Optional)

      Returns:
        The output of the discriminator.
      """
        # downsample
        ingress = self.conv0(x) 
        if illu is not None : ingress = ingress * (1 - illu * 2)
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)
        
        # print(out.shape, 'real_esrgan out shape')
        return out #if illu is None else out * illu