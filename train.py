import os
from os.path import join

import torch.backends.cudnn as cudnn
import faulthandler
import data.sirs_dataset as datasets
import util.util as util
from data.image_folder import read_fns
from engine import Engine
from options.net_options.train_options import TrainOptions
from tools import mutils
import data.new_dataset as datasets
import wandb
faulthandler.enable()

"""
This script is the main entry point for training a reflection removal model.

It performs the following key steps:
1.  **Parses Command-Line Options**: Uses `TrainOptions` to get all necessary
    configurations for the training process (e.g., learning rate, batch size,
    dataset paths, model parameters).
2.  **Initializes Datasets and DataLoaders**:
    -   Loads synthetic training data (VOC2012 based).
    -   Loads real-world training data (real and nature images).
    -   Creates a `FusionDataset` to combine these training sources based on
        specified ratios.
    -   Sets up `DataLoader` for the fused training data.
    -   Initializes multiple evaluation datasets (real20, solidobject, postcard, wild)
        and their corresponding DataLoaders.
3.  **Initializes Training Engine**: Creates an `Engine` instance which encapsulates
    the model, optimizer, and training/evaluation logic.
4.  **Handles Resuming and Initial Evaluation**: If the `--resume` or `--debug_eval`
    option is provided, it performs an initial evaluation on all test sets.
5.  **Defines Training Strategy**:
    -   Sets the GAN loss lambda to 0 (can be adjusted for different training phases).
    -   Sets the initial learning rate.
6.  **Main Training Loop**: Iterates for a predefined number of epochs (currently up to 20),
    calling `engine.train()` for each epoch with the fused training data.
    The `engine.train()` method itself handles per-iteration optimization, logging,
    and periodic evaluation based on frequencies defined in the options.
"""
opt = TrainOptions().parse()
cudnn.benchmark = True
opt.lambda_gan = 0
opt.display_freq = 1
opt.display_id = 1
opt.display_port = 8097
opt.display_freq = 1
opt.num_subnet = 4
if opt.debug:
    opt.display_id = 1
    opt.display_freq = 1
    opt.print_freq = 20
    opt.nEpochs = 40
    opt.max_dataset_size = 9999
    opt.no_log = False
    opt.nThreads = 0
    opt.decay_iter = 0
    opt.serial_batches = True
    opt.no_flip = True


datadir = os.path.join(os.path.expanduser('~'), '/opt/datasets/sirs')
datadir_syn = join(datadir, 'train/VOCdevkit/VOC2012/PNGImages')
datadir_real = join(datadir, 'train/real')
datadir_nature = join(datadir, 'train/nature')

train_dataset = datasets.DSRDataset(
    datadir_syn, read_fns('data/VOC2012_224_train_png.txt'), size=opt.max_dataset_size, enable_transforms=True)

train_dataset_real = datasets.DSRTestDataset(datadir_real, enable_transforms=True, if_align=opt.if_align)
train_dataset_nature = datasets.DSRTestDataset(datadir_nature, enable_transforms=True, if_align=opt.if_align)

train_dataset_fusion = datasets.FusionDataset([train_dataset,
                                               train_dataset_real,
                                               train_dataset_nature], [opt.dataset, (1-opt.dataset)/2, (1-opt.dataset)/2])

train_dataloader_fusion = datasets.DataLoader(
    train_dataset_fusion, batch_size=opt.batchSize, shuffle=not opt.serial_batches,
    num_workers=16, pin_memory=True)

eval_dataset_real = datasets.DSRTestDataset(join(datadir, f'test/real20_{opt.real20_size}'),
                                            fns=read_fns('data/real_test.txt'), if_align=opt.if_align)
eval_dataset_solidobject = datasets.DSRTestDataset(join(datadir, 'test/SIR2/SolidObjectDataset'),
                                                   if_align=opt.if_align)
eval_dataset_postcard = datasets.DSRTestDataset(join(datadir, 'test/SIR2/PostcardDataset'), if_align=opt.if_align)
eval_dataset_wild = datasets.DSRTestDataset(join(datadir, 'test/SIR2/WildSceneDataset'), if_align=opt.if_align)

eval_dataloader_real = datasets.DataLoader(
    eval_dataset_real, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataloader_solidobject = datasets.DataLoader(
    eval_dataset_solidobject, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)
eval_dataloader_postcard = datasets.DataLoader(
    eval_dataset_postcard, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataloader_wild = datasets.DataLoader(
    eval_dataset_wild, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

"""Main Loop"""
engine = Engine(opt,eval_dataloader_real,eval_dataloader_solidobject,eval_dataloader_postcard,eval_dataloader_wild)
result_dir = os.path.join(f'./experiment/{opt.name}/results',
                          mutils.get_formatted_time())


def set_learning_rate(lr):
  """Sets the learning rate for all optimizers in the model.

  Args:
    lr (float): The new learning rate to set.
  """
    for optimizer in engine.model.optimizers:
        print('[i] set learning rate to {}'.format(lr))
        util.set_opt_param(optimizer, 'lr', lr)


if opt.resume or opt.debug_eval:
    save_dir = os.path.join(result_dir, '%03d' % engine.epoch)
    os.makedirs(save_dir, exist_ok=True)
    engine.eval(eval_dataloader_real, dataset_name='testdata_real20', savedir=save_dir, suffix='real20')
    engine.eval(eval_dataloader_solidobject, dataset_name='testdata_solidobject', savedir=save_dir,
                suffix='solidobject')
    engine.eval(eval_dataloader_postcard, dataset_name='testdata_postcard', savedir=save_dir, suffix='postcard')
    engine.eval(eval_dataloader_wild, dataset_name='testdata_wild', savedir=save_dir, suffix='wild')

# define training strategy
engine.model.opt.lambda_gan = 0
set_learning_rate(opt.lr)
while engine.epoch <= 20:
    engine.train(train_dataloader_fusion)
