
from skimage.morphology import remove_small_objects
import numpy as np
import torch.utils.data
import random
from imgaug import augmenters as iaa
from misc.utils import cropping_center
from dataloader.augs import fix_mirror_padding
from .augs import (
    add_to_brightness,
    add_to_contrast,
    add_to_hue,
    add_to_saturation,
    gaussian_blur,
    median_blur,
)
import sys
sys.path.append('../')
from models.senucls.utils import get_bboxes_skimage as get_bboxes
# from models.senucls.utils import get_bboxes

def collate_fn(batch:list):
    new_batch = {}

    new_batch['img'] = torch.stack([b['img'] for b in batch], dim=0)
    new_batch['inst_map'] = torch.stack([b['inst_map'] for b in batch], dim=0)
    new_batch['tp_map'] = torch.stack([b['tp_map'] for b in batch], dim=0)
    new_batch['np_map'] = torch.stack([b['np_map'] for b in batch], dim=0)

    new_batch['batch_select_shape_feats'] = [b['select_shape_feats'] for b in batch]
    new_batch['batch_length'] = [b['length'] for b in batch]
    new_batch['batch_boxes'] = [b['boxes'] for b in batch]
    new_batch['batch_centers'] = [b['centers'] for b in batch]
    new_batch['batch_edge_points'] = [b['edge_points']for b in batch]
    new_batch['batch_edge_indexs'] = [b['edge_indexs'] for b in batch]
    new_batch['batch_pos_emb'] = [b['pos_emb'] for b in batch]
    new_batch['inst_classes'] = [b['inst_classes'] for b in batch]
    
    return new_batch

####
class FileLoader(torch.utils.data.Dataset):
    """Data Loader. Loads images from a file list and 
    performs augmentation with the albumentation library.
    After augmentation, horizontal and vertical maps are 
    generated.

    Args:
        file_list: list of filenames to load
        input_shape: shape of the input [h,w] - defined in config.py
        mask_shape: shape of the output [h,w] - defined in config.py
        mode: 'train' or 'valid'
        
    """

    # TODO: doc string

    def __init__(
        self,
        file_list,
        only_epithelial:bool,
        edge_num:int,
        point_num:int,
        with_type=False,
        input_shape=None,
        mask_shape=None,
        mode="train",
        setup_augmentor=True,
        target_gen=None,
    ):
        assert input_shape is not None and mask_shape is not None
        self.mode = mode
        self.only_epithelial = only_epithelial
        self.edge_num = edge_num
        self.point_num = point_num
        self.info_list = file_list
        self.with_type = with_type
        self.mask_shape = mask_shape
        self.input_shape = input_shape
        self.id = 0
        self.target_gen_func = target_gen[0]
        self.target_gen_kwargs = target_gen[1]
        if setup_augmentor:
            self.setup_augmentor(0, 0)
        return

    def setup_augmentor(self, worker_id, seed):
        self.augmentor = self.__get_augmentation(self.mode, seed)
        self.shape_augs = iaa.Sequential(self.augmentor[0])
        self.input_augs = iaa.Sequential(self.augmentor[1])
        self.id = self.id + worker_id
        return

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, idx):
        path = self.info_list[idx]
        data = np.load(path, allow_pickle=True)[..., :5]

        # split stacked channel into image and label
        img = (data[..., :3]).astype("uint8")  # RGB images
        ann = (data[..., 3:]).astype("int32")  # instance ID map and type map

        if self.shape_augs is not None:
            shape_augs = self.shape_augs.to_deterministic()
            img = shape_augs.augment_image(img)
            ann = shape_augs.augment_image(ann).copy()

        if self.input_augs is not None:
            input_augs = self.input_augs.to_deterministic()
            img = input_augs.augment_image(img)

        img = cropping_center(img, self.input_shape)
        # feed_dict = {"img": torch.from_numpy(img)}

        inst_map = cropping_center(ann[..., 0], self.mask_shape)  # HW1 -> HW
        inst_map = torch.from_numpy(remove_small_objects(fix_mirror_padding(inst_map), min_size=10))
        # inst_map = torch.from_numpy(label(fix_mirror_padding(inst_map).copy()>0))
        
        if self.with_type:
            type_map = torch.from_numpy(cropping_center(ann[..., 1], self.mask_shape))

        if self.only_epithelial:
            inst_map *= (type_map != 0)

        obj_ids = np.unique(inst_map)
        while len(obj_ids) == 1:
            #print(obj_ids)
            #print(path)
            new_idx = random.randint(0,len(self.info_list)-1)
            path = self.info_list[new_idx]
            #print(path)
            data = np.load(path, allow_pickle=True)[..., :5]

            # split stacked channel into image and label
            img = (data[..., :3]).astype("uint8")  # RGB images
            ann = (data[..., 3:]).astype("int32")  # instance ID map and type map

            if self.shape_augs is not None:
                shape_augs = self.shape_augs.to_deterministic()
                img = shape_augs.augment_image(img)
                ann = shape_augs.augment_image(ann).copy()

            if self.input_augs is not None:
                input_augs = self.input_augs.to_deterministic()
                img = input_augs.augment_image(img)

            img = cropping_center(img, self.input_shape)
            # feed_dict = {"img": torch.from_numpy(img)}

            # inst_map =  ann[..., 0]# HW1 -> HW

            # inst_map = torch.from_numpy(label(fix_mirror_padding(inst_map).copy()>0))
            inst_map = torch.from_numpy((remove_small_objects(fix_mirror_padding(cropping_center(ann[..., 0], self.mask_shape)), min_size=10)))

            if self.with_type:
                type_map = torch.from_numpy(cropping_center(ann[..., 1], self.mask_shape))

            if self.only_epithelial:
                inst_map *= (type_map != 0)

            obj_ids = np.unique(inst_map)

        # TODO: document hard coded assumption about #input
        np_map = self.target_gen_func(
            inst_map, self.mask_shape, **self.target_gen_kwargs)
        # feed_dict.update(target_dict)

        if not self.only_epithelial:
            type_map += np_map
        
        boxes, inst_class, centers, edge_points, edge_index, pos_emb, select_shape_feats = get_bboxes(inst_map, type_map, self.edge_num, self.point_num)
        
        feed_dict = {
            "img": torch.from_numpy(img),
            "inst_map": inst_map,
            "np_map": np_map,
            "tp_map": type_map,
            "select_shape_feats": select_shape_feats,
            "length": centers.shape[0],
            "boxes": boxes,
            "centers": centers,
            "edge_points": edge_points,
            "edge_indexs": edge_index,
            "pos_emb": pos_emb,
            "inst_classes": inst_class.reshape(inst_class.shape[0],1)
            }

        return feed_dict

    def __get_augmentation(self, mode, rng):
        if mode == "train":
            shape_augs = [
                # * order = ``0`` -> ``cv2.INTER_NEAREST``
                # * order = ``1`` -> ``cv2.INTER_LINEAR``
                # * order = ``2`` -> ``cv2.INTER_CUBIC``
                # * order = ``3`` -> ``cv2.INTER_CUBIC``
                # * order = ``4`` -> ``cv2.INTER_CUBIC``
                # ! for pannuke v0, no rotation or translation, just flip to avoid mirror padding
                iaa.Affine(
                    # scale images to 80-120% of their size, individually per axis
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # translate by -A to +A percent (per axis)
                    translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
                    shear=(-5, 5),  # shear by -5 to +5 degrees
                    rotate=(-179, 179),  # rotate by -179 to +179 degrees
                    order=0,  # use nearest neighbour
                    backend="cv2",  # opencv for fast processing
                    seed=rng,
                ),
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="center"
                ),
                iaa.Fliplr(0.5, seed=rng),
                iaa.Flipud(0.5, seed=rng),
            ]

            input_augs = [
                iaa.OneOf(
                    [
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: gaussian_blur(*args, max_ksize=3),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: median_blur(*args, max_ksize=3),
                        ),
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                        ),
                    ]
                ),
                iaa.Sequential(
                    [
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_hue(*args, range=(-8, 8)),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_saturation(
                                *args, range=(-0.2, 0.2)
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_brightness(
                                *args, range=(-26, 26)
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_contrast(
                                *args, range=(0.75, 1.25)
                            ),
                        ),
                    ],
                    random_order=True,
                ),
            ]
        elif mode == "valid":
            shape_augs = [
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="center"
                )
            ]
            input_augs = []

        return shape_augs, input_augs
