import torch.nn as nn

def make_layers(cfg, batch_norm=False):
  """Creates a sequence of layers based on a configuration.

  Args:
    cfg: A list defining the layer configuration. 'M' denotes a MaxPool2d layer,
         and integers denote Conv2d layers with that number of output channels.
    batch_norm: Whether to include BatchNorm2d layers after Conv2d layers.

  Returns:
    An nn.Sequential module containing the created layers.
  """
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

        
class VGG(nn.Module):
  """VGG-style network.

  Args:
    features: An nn.Sequential module representing the network features.
  """
    def __init__(self,features):
        super(VGG, self).__init__()
        self.features = features

    def forward(self, x):
      """Forward pass for the VGG network.

      Args:
        x: The input tensor.

      Returns:
        The output tensor.
      """
        x = self.features(x)

def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
  """Helper function to create a VGG model.

  Args:
    arch: The VGG architecture (e.g., 'vgg19').
    cfg: The configuration key for the VGG layers.
    batch_norm: Whether to use batch normalization.
    pretrained: Whether to load pretrained weights. (Not used in this function)
    progress: Whether to display a progress bar when downloading pretrained weights. (Not used)
    **kwargs: Additional keyword arguments for the VGG model.

  Returns:
    A VGG model.
  """
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model

def encoder(pretrained=False, progress=True, **kwargs):
  """Creates a VGG-style encoder.

  Args:
    pretrained: Whether to load pretrained weights. (Not used in this function)
    progress: Whether to display a progress bar when downloading pretrained weights. (Not used)
    **kwargs: Additional keyword arguments for the VGG model.

  Returns:
    A VGG-style encoder model.
  """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)