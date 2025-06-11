import os
import torch
import util.util as util


class BaseModel:
  """Base class for all models."""
    def name(self):
      """Returns the name of the model.

      Returns:
        The name of the model in lowercase.
      """
        return self.__class__.__name__.lower()

    def initialize(self, opt):
      """Initializes the model.

      Args:
        opt: The options for the model.
      """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        last_split = opt.checkpoints_dir.split('/')[-1]
        if opt.resume and last_split != 'checkpoints' and (last_split != opt.name or opt.supp_eval):

            self.save_dir = opt.checkpoints_dir
            self.model_save_dir = os.path.join(opt.checkpoints_dir.replace(opt.checkpoints_dir.split('/')[-1], ''),
                                               opt.name)
        else:
            self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
            self.model_save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self._count = 0

    def set_input(self, input):
      """Sets the input for the model.

      Args:
        input: The input data.
      """
        self.input = input

    def forward(self, mode='train'):
      """Performs a forward pass through the model.

      Args:
        mode: The mode of the forward pass (e.g., 'train', 'test').
      """
        pass

    # used in test time, no backprop
    def test(self):
      """Performs a forward pass in test mode.

      This method should not perform backpropagation.
      """
        pass

    def get_image_paths(self):
      """Returns the paths of the images used by the model."""
        pass

    def optimize_parameters(self):
      """Optimizes the model parameters."""
        pass

    def get_current_visuals(self):
      """Returns the current visuals for display.

      Returns:
        The current input data.
      """
        return self.input

    def get_current_errors(self):
      """Returns the current errors.

      Returns:
        An empty dictionary.
      """
        return {}

    def print_optimizer_param(self):
      """Prints the parameters of the last optimizer."""
        print(self.optimizers[-1])

    def save(self, label=None):
      """Saves the model to a file.

      Args:
        label: An optional label for the saved model.
      """
        epoch = self.epoch
        iterations = self.iterations

        if label is None:
            model_name = os.path.join(self.model_save_dir, self.opt.name + '_%03d_%08d.pt' % ((epoch), (iterations)))
        else:
            model_name = os.path.join(self.model_save_dir, self.opt.name + '_' + label + '.pt')

        torch.save(self.state_dict(), model_name)

    def save_eval(self, label=None):
      """Saves the model for evaluation.

      Args:
        label: An optional label for the saved model.
      """
        model_name = os.path.join(self.model_save_dir, label + '.pt')

        torch.save(self.state_dict_eval(), model_name)

    def _init_optimizer(self, optimizers):
      """Initializes the optimizers.

      Args:
        optimizers: A list of optimizers to initialize.
      """
        self.optimizers = optimizers
        for optimizer in self.optimizers:
            util.set_opt_param(optimizer, 'initial_lr', self.opt.lr)
            util.set_opt_param(optimizer, 'weight_decay', self.opt.wd)
