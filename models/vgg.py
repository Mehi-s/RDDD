from collections import namedtuple

import torch
from torchvision import models


class Vgg16(torch.nn.Module):
  """VGG16 model for feature extraction.

  Args:
    requires_grad: Whether to require gradients for the model parameters.
  """
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
      """Performs a forward pass through the Vgg16 model.

      Args:
        X: The input tensor.

      Returns:
        A named tuple containing the VGG16 features from different layers (relu1_2, relu2_2, relu3_3, relu4_3).
      """
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


class Vgg19(torch.nn.Module):
  """VGG19 model for feature extraction.

  Args:
    requires_grad: Whether to require gradients for the model parameters.
  """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        self.vgg_pretrained_features = models.vgg19(pretrained=True).features

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X, indices=None):
      """Performs a forward pass through the Vgg19 model.

      Args:
        X: The input tensor.
        indices: A list of layer indices to extract features from.
                 If None, default indices [2, 7, 12, 21, 30] will be used.

      Returns:
        A list of VGG19 features from the specified layers.
      """
        if indices is None:
            indices = [2, 7, 12, 21, 30]
        out = []
        for i in range(indices[-1]):
            X = self.vgg_pretrained_features[i](X)
            if (i + 1) in indices:
                out.append(X)

        return out


if __name__ == '__main__':
    vgg = Vgg19()
    import ipdb

    ipdb.set_trace()
