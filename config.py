import importlib
import random
import os
import cv2
import numpy as np

from dataset import get_dataset
from misc.utils import parse_json_file

# training class weights for each fold
cv0_weights = [[2.1787094536420635, 5.914368879138527, 2.6886567384700960], [3.1997485936058663, 1.4545974039518688]]
cv1_weights = [[2.2867743782898400, 5.396119679722524, 2.6498176180613044], [3.0364117299780453, 1.4910598310150085]]
cv2_weights = [[2.1567037964524830, 5.042323087721249, 2.9585082796899130], [2.7043464513304470, 1.5867351671483108]]

class Config(object):
    """Configuration file."""

    def __init__(self, args):
        self.seed = 10

        self.logging = True

        # turn on debug flag to trace some parallel processing problems more easily
        self.debug = False

        model_name = "senucls"
        model_mode = "original" # choose either `original` or `fast`

        if model_mode not in ["original", "fast"]:
            raise Exception("Must use either `original` or `fast` as model mode")

        # nr_type =  3 # number of nuclear types (including background)

        # whether to predict the nuclear type, availability depending on dataset!
        self.type_classification = True

        # shape information - 
        # below config is for original mode. 

        # Original implementation uses 256x256 for 20x datasets and 512x512 for 40x datasets
        aug_shape = [540, 540] # patch shape used during augmentation (larger patch may have less border artefacts)
        act_shape = [256, 256] # patch shape used as input to network - central crop performed after augmentation
        out_shape = [256, 256] # patch shape at output of network


        self.dataset_name = args.dataset_name # extracts dataset info from dataset.py
        self.source_dir = args.source_dir # path to project folder
        self.log_dir = args.save_dir # where checkpoints will be saved
        self.only_epithelial = args.only_epithelial # whether to train only on epithelial cells
        self.point_num = args.point_num 
        self.edge_num = args.edge_num 

        # paths to training and validation patches
        self.train_dir_list = [os.path.join(self.source_dir, tile['patches_dir']) for tile in parse_json_file(args.train_file)]  
        self.valid_dir_list = [os.path.join(self.source_dir, tile['patches_dir']) for tile in parse_json_file(args.val_file)]
        class_weights = cv0_weights if 'cv0' in args.train_file else cv1_weights if'cv1' in args.train_file else cv2_weights  if' cv2' in args.train_file else [1 for __ in range(args.nr_type-1)]# CHAINGE THIS TO ARGUMENT
        if args.only_epithelial:
            class_weights = class_weights[1] if 'cv' in args.train_file else [1,1]
        else:
            class_weights = class_weights[0] if 'cv' in args.train_file else [1,1,1]
        self.shape_info = {
            "train": {"input_shape": act_shape, "mask_shape": out_shape,},
            "valid": {"input_shape": act_shape, "mask_shape": out_shape,},
        }

        self.save_best_only = args.save_best_only
        # self.early_stop_patience = args.early_stopping

        # * parsing config to the running state and set up associated variables
        self.dataset = get_dataset(self.dataset_name)

        module = importlib.import_module(
            "models.%s.opt" % model_name
        )
        self.model_config = module.get_config(args.nr_type, model_mode, class_weights, args.only_epithelial, args.lr, args.batch_size, args.nr_epochs, args.save_best_only, args.early_stopping)
        
