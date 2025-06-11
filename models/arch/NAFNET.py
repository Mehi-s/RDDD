# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
# from .models.archs.arch_util import LayerNorm2d
import sys
sys.path.append('/ghome/zhuyr/Deref_RW/networks/')

class LayerNormFunction(torch.autograd.Function):
  """Custom Layer Normalization function.

  This function is used to implement LayerNorm2d.
  """

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
      """Forward pass for Layer Normalization.

      Args:
        ctx: Context object to save tensors for backward pass.
        x: Input tensor.
        weight: Weight tensor.
        bias: Bias tensor.
        eps: Epsilon value for numerical stability.

      Returns:
        Normalized tensor.
      """
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
      """Backward pass for Layer Normalization.

      Args:
        ctx: Context object with saved tensors from forward pass.
        grad_output: Gradient of the output.

      Returns:
        Gradients for input, weight, bias, and None for eps.
      """
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):
  """Layer Normalization for 2D inputs.

  Args:
    channels: Number of channels in the input tensor.
    eps: Epsilon value for numerical stability.
  """

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
      """Forward pass for LayerNorm2d.

      Args:
        x: Input tensor.

      Returns:
        Normalized tensor.
      """
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
  """Simple gate mechanism.

  Splits the input tensor into two halves along the channel dimension and multiplies them element-wise.
  """
    def forward(self, x):
      """Forward pass for SimpleGate.

      Args:
        x: Input tensor.

      Returns:
        Gated tensor.
      """
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
  """NAFNet block.

  Args:
    c: Number of channels.
    DW_Expand: Expansion factor for depth-wise convolution.
    FFN_Expand: Expansion factor for feed-forward network.
    drop_out_rate: Dropout rate.
  """
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
      """Forward pass for NAFBlock.

      Args:
        inp: Input tensor.

      Returns:
        Output tensor.
      """
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):
  """NAFNet model for image restoration.

  Args:
    img_channel: Number of input image channels.
    width: Width of the network.
    middle_blk_num: Number of blocks in the middle stage.
    enc_blk_nums: List of number of blocks in each encoder stage.
    dec_blk_nums: List of number of blocks in each decoder stage.
    global_residual: Whether to use global residual connection.
    drop_flag: Whether to use dropout.
    drop_rate: Dropout rate.
  """

    def __init__(self, img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28],
                 dec_blk_nums=[1, 1, 1, 1], global_residual = False, drop_flag = False, drop_rate = 0.4):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.global_residual = global_residual
        self.drop_flag = drop_flag

        if drop_flag:
            self.dropout = nn.Dropout2d(p=drop_rate)

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
      """Forward pass for NAFNet.

      Args:
        inp: Input tensor.

      Returns:
        Restored image tensor.
      """
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        base_inp = inp[:, :3, :, :]
        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        if self.drop_flag:
            x = self.dropout(x)

        x = self.ending(x)
        if self.global_residual:
            #print(x.shape, inp.shape, base_inp.shape)
            x = x + base_inp
        else:
            x
        return x[:, :, :H, :W]

    def check_image_size(self, x):
      """Pads the input image to be divisible by the padder size.

      Args:
        x: Input image tensor.

      Returns:
        Padded image tensor.
      """
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x



class NAFNet_wDetHead(nn.Module):
  """NAFNet model with a detection head for sparse reference.

  Args:
    img_channel: Number of input image channels.
    width: Width of the network.
    middle_blk_num: Number of blocks in the middle stage.
    enc_blk_nums: List of number of blocks in each encoder stage.
    dec_blk_nums: List of number of blocks in each decoder stage.
    global_residual: Whether to use global residual connection.
    drop_flag: Whether to use dropout.
    drop_rate: Dropout rate.
    concat: Whether to concatenate image features and sparse reference features.
    merge_manner: Manner of merging image features and sparse reference features.
                  0: Concatenate and convolve.
                  1: Add and convolve.
                  2: Multiply, add, and convolve.
  """

    def __init__(self, img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28],
                 dec_blk_nums=[1, 1, 1, 1], global_residual = False, drop_flag = False, drop_rate = 0.4,
                 concat = False, merge_manner = 0):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.global_residual = global_residual
        self.drop_flag = drop_flag
        self.concat = concat
        self.merge_manner = merge_manner

        if drop_flag:
            self.dropout = nn.Dropout2d(p=drop_rate)

        # --------------------------- Merge sparse & Img -------------------------------------------------------
        self.intro_Det = nn.Conv2d(in_channels=1, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.DetEnc = nn.Sequential( *[NAFBlock(width) for _ in range(3)] )
        if self.concat:
            self.Merge_conv = nn.Conv2d(in_channels=width *2 , out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        else:
            self.Merge_conv = nn.Conv2d(in_channels=width , out_channels=width, kernel_size=3, padding=1, stride=1,
                                        groups=1,
                                        bias=True)
        # ---------------------------  Merge sparse & Img -------------------------------------------------------

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, spare_ref):
      """Forward pass for NAFNet_wDetHead.

      Args:
        inp: Input image tensor.
        spare_ref: Sparse reference tensor.

      Returns:
        Restored image tensor.
      """
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        base_inp = inp #[:, :3, :, :]
        x = self.intro(inp)

        fea_sparse = self.DetEnc(self.intro_Det(spare_ref))

        if self.merge_manner ==0 and self.concat:
            x = torch.cat([x, fea_sparse], dim=1)
            x = self.Merge_conv(x)
        elif self.merge_manner == 1 and not self.concat:
            x = x + fea_sparse
            x = self.Merge_conv(x)
        elif self.merge_manner == 2 and not self.concat:
            x = x + fea_sparse *x
            x = self.Merge_conv(x)
        else:
            x = x
            print('Merge Flag Error!!!(No Merge Operation)    ---zyr 1031 ')

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        if self.drop_flag:
            x = self.dropout(x)

        x = self.ending(x)
        if self.global_residual:
            #print(x.shape, inp.shape, base_inp.shape)
            x = x + base_inp
        else:
            x
        return x[:, :, :H, :W]

    def check_image_size(self, x):
      """Pads the input image to be divisible by the padder size.

      Args:
        x: Input image tensor.

      Returns:
        Padded image tensor.
      """
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class NAFNet_refine(nn.Module):
  """NAFNet model for refinement.

  Args:
    img_channel: Number of input image channels (concatenated input and previous prediction).
    width: Width of the network.
    middle_blk_num: Number of blocks in the middle stage.
    enc_blk_nums: List of number of blocks in each encoder stage.
    dec_blk_nums: List of number of blocks in each decoder stage.
    global_residual: Whether to use global residual connection.
  """

    def __init__(self, img_channel=6, width=32, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28],
                 dec_blk_nums=[1, 1, 1, 1], global_residual = False):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.global_residual = global_residual

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, pre_pred):
      """Forward pass for NAFNet_refine.

      Args:
        inp: Input image tensor.
        pre_pred: Previous prediction tensor.

      Returns:
        Refined image tensor.
      """
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        pre_pred = self.check_image_size(pre_pred)

        network_in = torch.cat([inp, pre_pred ], dim= 1)

        x = self.intro(network_in)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)


        x = self.ending(x)
        if self.global_residual:

            x = x + inp[:3,:,:,:]
        else:
            x
        return x[:, :, :H, :W]

    def check_image_size(self, x):
      """Pads the input image to be divisible by the padder size.

      Args:
        x: Input image tensor.

      Returns:
        Padded image tensor.
      """
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


def print_param_number(net):
  """Prints the number of parameters in a network.

  Args:
    net: The input network.
  """
    print('#generator parameters:', sum(param.numel() for param in net.parameters()))
if __name__ == '__main__':
    img_channel = 3
    width = 32

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    # enc_blks = [1, 1, 1, 28]
    # middle_blk_num = 1
    # dec_blks = [1, 1, 1, 1]

    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]
    
    net = NAFNet_wDetHead(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks,global_residual = True,
                        concat= True, merge_manner= 2) #.cuda()
    #print(net)
    size = 352
    input = torch.randn([1,3,128, 128])#.cuda()  inp_shape = (5, 3, 128, 128)
    spare = torch.randn([1,1,128, 128])
    print(net(input, spare).size())
    print_param_number(net)


    
    #net_local = NAFNetLocal()#.cuda()

    #print_param_number(net)
    # print(net_local(input).size())
    # inp_shape = (3, 256, 256)
    #
    # from ptflops import get_model_complexity_info
    #
    # macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)
    #
    # params = float(params[:-3])
    # macs = float(macs[:-4])
    #
    # print(macs, params)