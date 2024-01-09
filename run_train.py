"""run_train.py

Main HoVer-Net training script.

Usage:
  run_train.py [--gpu=<id>] [--view=<dset>]
  run_train.py (-h | --help)
  run_train.py --version

Options:
  -h --help       Show this string.
  --version       Show version.
  --gpu=<id>      Comma separated GPU list. [default: 0,1,2,3]
  --view=<dset>   Visualise images after augmentation. Choose 'train' or 'valid'.
"""

import cv2
import faulthandler


faulthandler.enable()
#cv2.setNumThreads(0)
import argparse
import glob
import importlib
import inspect
import json
import os
import shutil
from natsort import natsorted
import matplotlib
import numpy as np
import torch
from docopt import docopt
from tensorboardX import SummaryWriter
from torch.nn import DataParallel  # TODO: switch to DistributedDataParallel
# from torch.utils.data import DataLoader
from run_utils.utils import MultiEpochsDataLoader as DataLoader
# import torchcontrib
# from torchcontrib.optim import SWA
from config import Config
from dataloader.train_loader import FileLoader
from misc.utils import rm_n_mkdir, parse_json_file
from run_utils.engine import RunEngine
from run_utils.utils import (
    check_log_dir,
    check_manual_seed,
    colored,
    convert_pytorch_checkpoint,
    create_logger
)
import argparse


#### have to move outside because of spawn
# * must initialize augmentor per worker, else duplicated rng generators may happen
def worker_init_fn(worker_id):
    # ! to make the seed chain reproducible, must use the torch random, not numpy
    # the torch rng from main thread will regenerate a base seed, which is then
    # copied into the dataloader each time it created (i.e start of each epoch)
    # then dataloader with this seed will spawn worker, now we reseed the worker
    worker_info = torch.utils.data.get_worker_info()
    # to make it more random, simply switch torch.randint to np.randint
    worker_seed = torch.randint(0, 2 ** 32, (1,))[0].cpu().item() + worker_id
    # print('Loader Worker %d Uses RNG Seed: %d' % (worker_id, worker_seed))
    # retrieve the dataset copied into this worker process
    # then set the random seed for each augmentation
    worker_info.dataset.setup_augmentor(worker_id, worker_seed)
    return

def get_args():

    parser = argparse.ArgumentParser()

    ## Directory arguments
    parser.add_argument('--val_file', type=str)
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--save_dir', type=str, help='directory where the checkpoints after each epoch will be saved')
    parser.add_argument('--source_dir', type=str, default='/storage/homefs/jg23p152/project/', help='path to project folder (where data is stored)')

    ## Model training arguments
    parser.add_argument('--dataset_name', type=str, default='lizard')
    parser.add_argument('--nr_type', type=int, default=3, help='number of nuclear types (including background)')
    parser.add_argument('--lr', type=str, default='1e-4,1e-4', help='learning rate for each training phase separated by comma')
    parser.add_argument('--batch_size', type=str, default='10,4', help='batch size for each training phase separated by comma')
    parser.add_argument('--nr_epochs', type=str, default='50,50', help='number of epochs for each training phase separated by comma')
    parser.add_argument('--save_best_only', action='store_true', help='use if you only want the best epoch checkpoint to be saved')
    parser.add_argument('--early_stopping', type=int, default=1000, help='number of epochs to wait before stopping training if validation loss does not improve')
    parser.add_argument('--only_epithelial', action='store_true', help='use if you only want to train on epithelial cells')
    parser.add_argument('--edge_num', type=int, default=4, help='number of edges per node in the graph')
    parser.add_argument('--point_num', type=int, default=18, help='number of points per edge in the graph')

    ## GPU arguments
    parser.add_argument('--gpu', type=str, help='comma separated GPU list')
    parser.add_argument('--view', type=str, default=None, help='visualise images after augmentation. Choose `train` or `valid`')
    args = parser.parse_args()

    # turn batch_size, nr_epochs, lr into list
    args.lr = [float(lr) for lr in args.lr.split(',')]
    args.batch_size = [int(batch_size) for batch_size in args.batch_size.split(',')]
    args.nr_epochs = [int(nr_epochs) for nr_epochs in args.nr_epochs.split(',')]
    return args
####
class TrainManager(Config):
    """Either used to view the dataset or to initialise the main training loop."""

    def __init__(self, args):
        super().__init__(args)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        return

    ####
    def view_dataset(self, mode="train"):
        """
        Manually change to plt.savefig or plt.show 
        if using on headless machine or not
        """
        self.nr_gpus = 1
        import matplotlib.pyplot as plt
        check_manual_seed(self.seed)
        # TODO: what if each phase want diff annotation ?
        phase_list = self.model_config["phase_list"][0]
        target_info = phase_list["target_info"]
        prep_func, prep_kwargs = target_info["viz"]
        dataloader = self._get_datagen(2, mode, target_info["gen"])
        for batch_data in dataloader:  
            # convert from Tensor to Numpy
            batch_data = {k: v.numpy() for k, v in batch_data.items()}
            viz = prep_func(batch_data, is_batch=True, **prep_kwargs)
            plt.imshow(viz)
            plt.show()
        self.nr_gpus = -1
        return

    ####
    def _get_datagen(self, batch_size, run_mode, target_gen, nr_procs=0, fold_idx=0):
        nr_procs = nr_procs if not self.debug else 0

        # ! Hard assumption on file type
        file_list = []
        if run_mode == "train":
            data_dir_list = self.train_dir_list
        else:
            data_dir_list = self.valid_dir_list
        for dir_path in data_dir_list:
            file_list.extend(glob.glob("%s/*.npy" % dir_path))
        file_list = natsorted(file_list)  # to always ensure same input ordering

        assert len(file_list) > 0, (
            "No .npy found for `%s`, please check `%s` in `config.py`"
            % (run_mode, "%s_dir_list" % run_mode)
        )
        print("Dataset %s: %d" % (run_mode, len(file_list)))
        input_dataset = FileLoader(
            file_list,
            mode=run_mode,
            with_type=self.type_classification,
            setup_augmentor=nr_procs == 0,
            target_gen=target_gen,
            **self.shape_info[run_mode]
        )

        dataloader = DataLoader(
            input_dataset,
            num_workers=nr_procs,
            batch_size=batch_size * self.nr_gpus,
            shuffle=run_mode == "train",
            drop_last=run_mode == "train",
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )
        return dataloader

    ####
    def run_once(self, opt, run_engine_opt, log_dir, prev_log_dir=None, fold_idx=0):
        """Simply run the defined run_step of the related method once."""
        check_manual_seed(self.seed)

        log_info = {}
        continue_training = False
        if self.logging:
            # check_log_dir(log_dir)
            if not opt["continue_training"]:
                rm_n_mkdir(log_dir)
            elif not os.path.isfile(os.path.join(log_dir, 'net_last_epoch.tar')):
                rm_n_mkdir(log_dir)
            else:
                os.makedirs(log_dir, exist_ok=True)
                opt["run_info"]["net"]["pretrained"] = -1
                prev_log_dir = log_dir
                continue_training = True


            tfwriter = SummaryWriter(log_dir=log_dir)
            json_log_file = log_dir + "/stats.json"
            with open(json_log_file, "w") as json_file:
                json.dump({}, json_file)  # create empty file
            log_info = {
                "json_file": json_log_file,
                "tfwriter": tfwriter,
            }

        ####
        loader_dict = {}
        for runner_name, runner_opt in run_engine_opt.items():
            loader_dict[runner_name] = self._get_datagen(
                opt["batch_size"][runner_name],
                runner_name,
                opt["target_info"]["gen"],
                nr_procs=runner_opt["nr_procs"],
                fold_idx=fold_idx,
            )
        ####
        def get_last_chkpt_path(prev_phase_dir, net_name):
            if not self.save_best_only:
                stat_file_path = prev_phase_dir + "/stats.json"
                with open(stat_file_path) as stat_file:
                    info = json.load(stat_file)
                epoch_list = [int(v) for v in info.keys()]
                last_chkpts_path = "%s/%s_epoch=%d.tar" % (
                    prev_phase_dir,
                    net_name,
                    max(epoch_list),
                )
            else:
                last_chkpts_path = os.path.join(prev_phase_dir, "%s_last_epoch.tar" % net_name)
            return last_chkpts_path

        # TODO: adding way to load pretrained weight or resume the training
        # parsing the network and optimizer information
        net_run_info = {}
        net_info_opt = opt["run_info"]
        for net_name, net_info in net_info_opt.items():
            assert inspect.isclass(net_info["desc"]) or inspect.isfunction(
                net_info["desc"]
            ), "`desc` must be a Class or Function which instantiate NEW objects !!!"
            print(net_info) 
            net_desc = net_info["desc"]() # create_model (HoVerNet)

            # TODO: customize print-out for each run ?
            # summary_string(net_desc, (3, 270, 270), device='cpu')

            pretrained_path = net_info["pretrained"]
            if pretrained_path is not None:
                if pretrained_path == -1:
                    # * depend on logging format so may be broken if logging format has been changed
                    pretrained_path = get_last_chkpt_path(prev_log_dir, net_name)
                    net_state_dict = torch.load(pretrained_path)
                else:
                    chkpt_ext = os.path.basename(pretrained_path).split(".")[-1]
                    if chkpt_ext == "npz":
                        net_state_dict = dict(np.load(pretrained_path))
                        net_state_dict = {
                            k: torch.from_numpy(v) for k, v in net_state_dict.items()
                        }
                    elif chkpt_ext == "tar":  # ! assume same saving format we desire
                        net_state_dict = torch.load(pretrained_path)
                        #net_state_dict.pop('encoder.patch_embed1.proj.weight')
                        #net_state_dict.pop('encoder.patch_embed1.proj.bias')

                colored_word = colored(net_name, color="red", attrs=["bold"])
                print(
                    "Model `%s` pretrained path: %s" % (colored_word, pretrained_path)
                )

                # load_state_dict returns (missing keys, unexpected keys)
                global_epoch = net_state_dict['epoch'] if 'epoch' in net_state_dict else 0
                net_state_dict = convert_pytorch_checkpoint(net_state_dict["desc"])
                load_feedback = net_desc.load_state_dict(net_state_dict, strict=False)
                # * uncomment for your convenience
                print("Missing Variables: \n", load_feedback[0])
                print("Detected Unknown Variables: \n", load_feedback[1])

            # * extremely slow to pass this on DGX with 1 GPU, why (?)
            net_desc = DataParallel(net_desc)
            net_desc = net_desc.to(self.device)
            # print(net_desc) # * dump network definition or not?
            optimizer, optimizer_args = net_info["optimizer"]
            optimizer = optimizer(net_desc.parameters(), **optimizer_args)
            #####SWA###
            #optimizer = torchcontrib.optim.SWA(optimizer, swa_start=10, swa_freq=5, swa_lr=0.05)

            #optimizer.swap_swa_sgd()
 
            # training loop
            if continue_training:
                # option 1: load before first LR update
                if current_epoch <= net_info["scheduler_patience"]:
                    run_engine_opt["train"]["callbacks"][Events.EPOCH_COMPLETED][-1]._update_start_epoch(current_epoch,net_info["scheduler_patience"])
                
                # option 2: load between updates. Optimizer should be directly updated to the last LR and we need to know nr of epochs until next update
                else:
                    nr_updates = current_epoch // net_info["scheduler_patience"]
                    for g in optimizer.param_groups:
                        g['lr'] *= 0.1 ** nr_updates
                    run_engine_opt["train"]["callbacks"][Events.EPOCH_COMPLETED][-1]._update_start_epoch(current_epoch % net_info["scheduler_patience"],net_info["scheduler_patience"])
                # else:
                #     scheduler_epoch = current_epoch - opt["nr_epochs"]
            else:
                   run_engine_opt["train"]["callbacks"][Events.EPOCH_COMPLETED][-1]._update_start_epoch(0,net_info["scheduler_patience"])
            # else:
            #     scheduler_epoch = 0
            # TODO: expand for external aug for scheduler
            
            nr_iter = opt["nr_epochs"] * len(loader_dict["train"])
            scheduler = net_info["lr_scheduler"](optimizer)
            net_run_info[net_name] = {
                "desc": net_desc,
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                # TODO: standardize API for external hooks
                "extra_info": net_info["extra_info"],
            }

        # parsing the running engine configuration
        assert (
            "train" in run_engine_opt
        ), "No engine for training detected in description file"

        # initialize runner and attach callback afterward
        # * all engine shared the same network info declaration
        runner_dict = {}
        for runner_name, runner_opt in run_engine_opt.items():
            runner_dict[runner_name] = RunEngine(
                dataloader=loader_dict[runner_name],
                engine_name=runner_name,
                run_step=runner_opt["run_step"],
                run_info=net_run_info,
                log_info=log_info,
            )

        for runner_name, runner in runner_dict.items():
            callback_info = run_engine_opt[runner_name]["callbacks"]
            for event, callback_list, in callback_info.items():
                for callback in callback_list:
                    if callback.engine_trigger:
                        triggered_runner_name = callback.triggered_engine_name
                        callback.triggered_engine = runner_dict[triggered_runner_name]
                    runner.add_event_handler(event, callback)

        # retrieve main runner
        
        if current_epoch < opt["nr_epochs"]:
            main_runner = runner_dict["train"]
            main_runner.state.logging = self.logging
            main_runner.state.log_dir = log_dir
            main_runner.state.global_epoch = global_epoch
            main_runner.state.curr_epoch = current_epoch
            runner_dict["valid"].state.global_epoch = global_epoch
            # start the run loop
            main_runner.run(opt["nr_epochs"])

        print("\n")
        print("########################################################")
        print("########################################################")
        print("\n")
        return

    ####
    def run(self):
        """Define multi-stage run or cross-validation or whatever in here."""
        self.nr_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        print('Detect #GPUS: %d' % self.nr_gpus)

        phase_list = self.model_config["phase_list"]
        engine_opt = self.model_config["run_engine"]

        prev_save_path = None
        for phase_idx, phase_info in enumerate(phase_list):
            if len(phase_list) == 1:
                save_path = self.log_dir
            else:
                save_path = self.log_dir + "/%02d/" % (phase_idx)
            self.run_once(
                phase_info, engine_opt, save_path, prev_log_dir=prev_save_path
            )
            prev_save_path = save_path


####
if __name__ == "__main__":
    # args = docopt(__doc__, version="HoVer-Net v1.0")
    args = get_args()
    trainer = TrainManager(args)

    if args.view:
        if args.view != "train" and args.view != "valid":
            raise Exception('Use "train" or "valid" for --view.')
        trainer.view_dataset(args.view)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        trainer.run()
