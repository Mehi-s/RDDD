import torch.nn as nn
import timm
import torch
import torch.nn.functional as F
class PretrainedConvNext(nn.Module):
  """Pretrained ConvNext model for classification.

  Args:
    model_name: The name of the ConvNext model to use (e.g., 'convnext_base').
    pretrained: Whether to load pretrained weights. (Currently set to False in constructor)
  """
    def __init__(self, model_name='convnext_base', pretrained=True):
        super(PretrainedConvNext, self).__init__()
        # Load the pretrained ConvNext model from timm
        self.model = timm.create_model(model_name, pretrained=False, num_classes=0)
        self.head = nn.Linear(768, 6)
    def forward(self, x):
      """Forward pass for the PretrainedConvNext model.

      Args:
        x: The input tensor.

      Returns:
        The output of the classification head.
      """
        with torch.no_grad():
            cls_input = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        # Forward pass through the ConvNext model
        out = self.model(cls_input)
        out = self.head(out)
        # alpha, beta = out[..., :3].unsqueeze(-1).unsqueeze(-1),\
        #               out[..., 3:].unsqueeze(-1).unsqueeze(-1)

        #out = alpha * x + beta
        # print(out.shape)
        return out#alpha,beta#out #out[..., :3], out[..., 3:]
class PretrainedConvNext_e2e(nn.Module):
  """End-to-end Pretrained ConvNext model.

  This model applies the classification output (alpha, beta) to the input image.

  Args:
    model_name: The name of the ConvNext model to use (e.g., 'convnext_base').
    pretrained: Whether to load pretrained weights.
  """
    def __init__(self, model_name='convnext_base', pretrained=True):
        super(PretrainedConvNext_e2e, self).__init__()
        # Load the pretrained ConvNext model from timm
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.head = nn.Linear(768, 6)
    def forward(self, x):
      """Forward pass for the PretrainedConvNext_e2e model.

      Args:
        x: The input tensor.

      Returns:
        The output tensor after applying alpha and beta scaling and shifting.
      """
        with torch.no_grad():
            cls_input = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        # Forward pass through the ConvNext model
        out = self.model(cls_input)
        out = self.head(out)
        alpha, beta = out[..., :3].unsqueeze(-1).unsqueeze(-1),\
                      out[..., 3:].unsqueeze(-1).unsqueeze(-1)

        out = alpha * x + beta
        #print(out.shape)
        return out#alpha,beta#out #out[..., :3], out[..., 3:]

if __name__ == "__main__":
    model = PretrainedConvNext('convnext_small_in22k')
    print("Testing PretrainedConvNext model...")
    # Assuming a dummy input tensor of size (1, 3, 224, 224) similar to an image in the ImageNet dataset
    dummy_input = torch.randn(20, 3, 224, 224)
    output_x, output_y = model(dummy_input)
    print("Output shape:", output_x.shape)
    print("Test completed successfully.")
