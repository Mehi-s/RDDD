import torch


def attn_cosine_sim(x, eps=1e-08):
  """Computes the cosine similarity of attention maps.

  Args:
    x: The input tensor of attention maps.
    eps: A small epsilon value to prevent division by zero.

  Returns:
    The cosine similarity matrix.
  """
    assert x.shape[0] == 1, 'x.shape[0] must eqs 1'
    x = x[0]  # TEMP: getting rid of redundant dimension, TBF
    norm1 = x.norm(dim=2, keepdim=True)
    factor = torch.clamp(norm1 @ norm1.permute(0, 2, 1), min=eps)
    sim_matrix = (x @ x.permute(0, 2, 1)) / factor
    return sim_matrix


class VitExtractor:
  """Extracts features from Vision Transformer (ViT) models.

  Args:
    model_name: The name of the ViT model to use (e.g., 'dino_vits8').
    device: The device to run the model on (e.g., 'cuda', 'cpu').
  """
    BLOCK_KEY = 'block'
    ATTN_KEY = 'attn'
    PATCH_IMD_KEY = 'patch_imd'
    QKV_KEY = 'qkv'
    KEY_LIST = [BLOCK_KEY, ATTN_KEY, PATCH_IMD_KEY, QKV_KEY]

    def __init__(self, model_name, device):
      """Initializes the VitExtractor.

      Args:
        model_name: The name of the ViT model to use.
        device: The device to run the model on.
      """
        self.model = torch.hub.load('facebookresearch/dino:main', model_name).to(device)
        self.model.eval()
        self.model_name = model_name
        self.hook_handlers = []
        self.layers_dict = {}
        self.outputs_dict = {}
        for key in VitExtractor.KEY_LIST:
            self.layers_dict[key] = []
            self.outputs_dict[key] = []
        self._init_hooks_data()

    def _init_hooks_data(self):
      """Initializes the dictionaries for storing hooked layers and outputs."""
        self.layers_dict[VitExtractor.BLOCK_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.ATTN_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.QKV_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.PATCH_IMD_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for key in VitExtractor.KEY_LIST:
            # self.layers_dict[key] = kwargs[key] if key in kwargs.keys() else []
            self.outputs_dict[key] = []

    def _register_hooks(self, **kwargs):
      """Registers forward hooks to extract features from specified layers."""
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in self.layers_dict[VitExtractor.BLOCK_KEY]:
                self.hook_handlers.append(block.register_forward_hook(self._get_block_hook()))
            if block_idx in self.layers_dict[VitExtractor.ATTN_KEY]:
                self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_attn_hook()))
            if block_idx in self.layers_dict[VitExtractor.QKV_KEY]:
                self.hook_handlers.append(block.attn.qkv.register_forward_hook(self._get_qkv_hook()))
            if block_idx in self.layers_dict[VitExtractor.PATCH_IMD_KEY]:
                self.hook_handlers.append(block.attn.register_forward_hook(self._get_patch_imd_hook()))

    def _clear_hooks(self):
      """Removes all registered forward hooks."""
        for handler in self.hook_handlers:
            handler.remove()
        self.hook_handlers = []

    def _get_block_hook(self):
      """Returns a hook function for extracting block outputs."""
        def _get_block_output(model, input, output):
            self.outputs_dict[VitExtractor.BLOCK_KEY].append(output)

        return _get_block_output

    def _get_attn_hook(self):
      """Returns a hook function for extracting attention outputs."""
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.ATTN_KEY].append(output)

        return _get_attn_output

    def _get_qkv_hook(self):
      """Returns a hook function for extracting QKV outputs."""
        def _get_qkv_output(model, inp, output):
            self.outputs_dict[VitExtractor.QKV_KEY].append(output)

        return _get_qkv_output

    # TODO: CHECK ATTN OUTPUT TUPLE
    def _get_patch_imd_hook(self):
      """Returns a hook function for extracting intermediate patch outputs from attention blocks."""
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.PATCH_IMD_KEY].append(output[0])

        return _get_attn_output

    def get_feature_from_input(self, input_img):  # List([B, N, D])
      """Extracts block features from the input image.

      Args:
        input_img: The input image tensor.

      Returns:
        A list of block features.
      """
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.BLOCK_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_qkv_feature_from_input(self, input_img):
      """Extracts QKV features from the input image.

      Args:
        input_img: The input image tensor.

      Returns:
        A list of QKV features.
      """
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.QKV_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_attn_feature_from_input(self, input_img):
      """Extracts attention features from the input image.

      Args:
        input_img: The input image tensor.

      Returns:
        A list of attention features.
      """
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.ATTN_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_patch_size(self):
      """Returns the patch size of the ViT model.

      Returns:
        The patch size (8 or 16).
      """
        return 8 if "8" in self.model_name else 16

    def get_width_patch_num(self, input_img_shape):
      """Calculates the number of patches along the width of the input image.

      Args:
        input_img_shape: The shape of the input image (B, C, H, W).

      Returns:
        The number of patches along the width.
      """
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return w // patch_size

    def get_height_patch_num(self, input_img_shape):
      """Calculates the number of patches along the height of the input image.

      Args:
        input_img_shape: The shape of the input image (B, C, H, W).

      Returns:
        The number of patches along the height.
      """
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return h // patch_size

    def get_patch_num(self, input_img_shape):
      """Calculates the total number of patches in the input image.

      Args:
        input_img_shape: The shape of the input image (B, C, H, W).

      Returns:
        The total number of patches.
      """
        patch_num = 1 + (self.get_height_patch_num(input_img_shape) * self.get_width_patch_num(input_img_shape))
        return patch_num

    def get_head_num(self):
      """Returns the number of attention heads in the ViT model.

      Returns:
        The number of attention heads.
      """
        if "dino" in self.model_name:
            return 6 if "s" in self.model_name else 12
        return 6 if "small" in self.model_name else 12

    def get_embedding_dim(self):
      """Returns the embedding dimension of the ViT model.

      Returns:
        The embedding dimension.
      """
        if "dino" in self.model_name:
            return 384 if "s" in self.model_name else 768
        return 384 if "small" in self.model_name else 768

    def get_queries_from_qkv(self, qkv, input_img_shape):
      """Extracts queries from the QKV tensor.

      Args:
        qkv: The QKV tensor.
        input_img_shape: The shape of the input image (B, C, H, W).

      Returns:
        The queries tensor.
      """
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        q = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[0]
        return q

    def get_keys_from_qkv(self, qkv, input_img_shape):
      """Extracts keys from the QKV tensor.

      Args:
        qkv: The QKV tensor.
        input_img_shape: The shape of the input image (B, C, H, W).

      Returns:
        The keys tensor.
      """
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        k = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[1]
        return k

    def get_values_from_qkv(self, qkv, input_img_shape):
      """Extracts values from the QKV tensor.

      Args:
        qkv: The QKV tensor.
        input_img_shape: The shape of the input image (B, C, H, W).

      Returns:
        The values tensor.
      """
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        v = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[2]
        return v

    def get_keys_from_input(self, input_img, layer_num):
      """Extracts keys from a specific layer of the ViT model.

      Args:
        input_img: The input image tensor.
        layer_num: The layer number to extract keys from.

      Returns:
        The keys tensor from the specified layer.
      """
        qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
        keys = self.get_keys_from_qkv(qkv_features, input_img.shape)
        return keys

    def get_keys_self_sim_from_input(self, input_img, layer_num):
      """Computes the self-similarity of keys from a specific layer of the ViT model.

      Args:
        input_img: The input image tensor.
        layer_num: The layer number to extract keys from.

      Returns:
        The self-similarity map of keys.
      """
        keys = self.get_keys_from_input(input_img, layer_num=layer_num)
        h, t, d = keys.shape
        concatenated_keys = keys.transpose(0, 1).reshape(t, h * d)
        ssim_map = attn_cosine_sim(concatenated_keys[None, None, ...])
        return ssim_map