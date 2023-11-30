import glob
import cv2
import numpy as np
import scipy.io as sio


class __AbstractDataset(object):
    """Abstract class for interface of subsequent classes.
    Main idea is to encapsulate how each dataset should parse
    their images and annotations.
    
    """

    def load_img(self, path):
        raise NotImplementedError

    def load_ann(self, path, with_type=False):
        raise NotImplementedError


####
class __Kumar(__AbstractDataset):
    """Defines the Kumar dataset as originally introduced in:

    Kumar, Neeraj, Ruchika Verma, Sanuj Sharma, Surabhi Bhargava, Abhishek Vahadane, 
    and Amit Sethi. "A dataset and a technique for generalized nuclear segmentation for 
    computational pathology." IEEE transactions on medical imaging 36, no. 7 (2017): 1550-1560.

    """

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        # assumes that ann is HxW
        assert not with_type, "Not support"
        ann_inst = sio.loadmat(path)["inst_map"]
        ann_inst = ann_inst.astype("int32")
        ann = np.expand_dims(ann_inst, -1)
        return ann

    def load_mask(self, path,with_type=False):
        # assumes that ann is HxW
        #assert not with_type, "Not support"
        ann_inst = cv2.imread(path)
        ann_inst = ann_inst.astype("int32")//255
        if np.isnan(np.min(ann_inst)):
            print('wrong value')
        else:
            return ann_inst

####
class __CPM17(__AbstractDataset):
    """Defines the CPM 2017 dataset as originally introduced in:

    Vu, Quoc Dang, Simon Graham, Tahsin Kurc, Minh Nguyen Nhat To, Muhammad Shaban, 
    Talha Qaiser, Navid Alemi Koohbanani et al. "Methods for segmentation and classification 
    of digital microscopy tissue images." Frontiers in bioengineering and biotechnology 7 (2019).

    """

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        assert not with_type, "Not support"
        # assumes that ann is HxW
        ann_inst = sio.loadmat(path)["inst_map"]
        ann_inst = ann_inst.astype("int32")
        ann = np.expand_dims(ann_inst, -1)
        return ann


####
class __CoNSeP(__AbstractDataset):
    """Defines the CoNSeP dataset as originally introduced in:

    Graham, Simon, Quoc Dang Vu, Shan E. Ahmed Raza, Ayesha Azam, Yee Wah Tsang, Jin Tae Kwak, 
    and Nasir Rajpoot. "Hover-Net: Simultaneous segmentation and classification of nuclei in 
    multi-tissue histology images." Medical Image Analysis 58 (2019): 101563
    
    """

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        # assumes that ann is HxW
        ann_inst = sio.loadmat(path)["inst_map"]
        if with_type:
            ann_type = sio.loadmat(path)["type_map"]

            # merge classes for CoNSeP (in paper we only utilise 3 nuclei classes and background)
            # If own dataset is used, then the below may need to be modified
            ann_type[(ann_type == 3) | (ann_type == 4)] = 3
            ann_type[(ann_type == 5) | (ann_type == 6) | (ann_type == 7)] = 4

            ann = np.dstack([ann_inst, ann_type])
            ann = ann.astype("int32")
        else:
            ann = np.expand_dims(ann_inst, -1)
            ann = ann.astype("int32")

        return ann
    

####    
class __Lizard(__AbstractDataset):
    """Defines the Lizard dataset sample used for epithelial cell classification into malignant and healthy instances.
    The Lizard Dataset was fist introduced in https://doi.org/10.1109/ICCVW54120.2021.00082
    Lizard Dataset differenciates between 6 cell types:
        1. Neutrophil
        2. Epithelial
        3. Lymphocyte
        4. Plasma
        5. Eosinophil
        6. Connective
       (0. Background)
    """
    def __init__(self):
        self.inst_map_key = 'inst_map'
        self.cell_type_key = 'class'

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    
    def _inst_to_class(self, inst_map:np.ndarray, types:np.ndarray, to_keep:list=[2]) -> tuple:
        """Takes an instance map and a cell type list to create a 'type' map. Only the cell types included in to_keep
        will be preserved, the rest will be merged into the background class.

        Args:
            inst_map (np.ndarray): Instance map with cell instances numbered from 1 to cell_count
            types (np.ndarray): array.shape = (cell_count, 1), with same order as in 'inst_map'. Contains cell types of instances
            to_keep (list, optional): List of cell types to keep. Defaults to [2].

        Returns:
            np.ndarray: cell type map of same shape as 'inst_map'
            np.ndarray: inst_map of same shape as original 'inst_map' containing only instances of cell types in to_keep
        """

        types = types.flatten()
        type_map = np.zeros(inst_map.shape)
        keep_inst_map = np.zeros(inst_map.shape)

        for cell_type in to_keep:
            pos = list(np.where(types==cell_type)[0]+1)
            bool_map = np.isin(inst_map, pos)
            keep_inst_map += bool_map
            type_map += bool_map * cell_type

        # Filter out all instances that don't belong to the cell types in to_keep
        inst_map = inst_map * keep_inst_map 
        return type_map, inst_map

    def load_ann(self, path, with_type=False, to_keep:list=[2], is_healthy:bool=True):
        # in this case the annotation is saved as a dictionary with keys:
            # 'inst_map': instance map with cell instances numbered from 1 to cell_count
            # 'class': array of array.shape = (cell_count, 1), with same order as in 'inst_map'

        mat_file = sio.loadmat(path)
        ann_inst = mat_file[self.inst_map_key]

        # In our task we only care about epithelial cells, so we merge all other cell types into the background class
        # Consequently we have three classes in the annotation mask from the original six:
        #  0. Background
        #  1. Healthy Epithelial - if source image in 'tiles_healthy' folder
        #  2. Malignant Epithelial - if source image in 'tiles_malignant' folder
        # We specify the that the only class to be kept in the annotation map is the epithelial class (2)
        
        if with_type:

            ann_type, __ = self._inst_to_class(ann_inst, mat_file[self.cell_type_key], to_keep=to_keep)
            ann_type[(ann_type == 2)] = 1 if is_healthy else 2
            
            # Stack instance map and class map to create a single annotation map 
            ann = np.dstack([ann_inst, ann_type])
            ann = ann.astype("int32")
            return ann
            
        else:
            ann = np.expand_dims(ann_inst, -1)
            ann = ann.astype("int32")

        return ann

    

####
class __PanNuke(__Lizard):
    def __init__(self):
        self.inst_map_key = 'inst_map'
        self.cell_type_key = 'class'


####
class __TCGA(__Lizard):
    def __init__(self):
        self.inst_map_key = 'inst_map'
        self.cell_centroid = 'inst_centroid'
        self.im_shape = []
    
    def load_img(self, path):
        im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        self.im_shape = im.shape
        return im
    
    def load_ann(self, path, with_type=False, to_keep:list=[2], is_healthy:bool=True):
        return np.zeros(self.im_shape[:2]+(2,), dtype=np.int32)
        

        Warning("TCGA dataset is not yet supported")
    ##TODO: the TCGA annotation is limited to cell centroids. Need to run segmentation before it can be used.


####
def get_dataset(name):
    """Return a pre-defined dataset object associated with `name`."""
    name_dict = {
        "kumar": lambda: __Kumar(),
        "cpm17": lambda: __CPM17(),
        "consep": lambda: __CoNSeP(),
        "lizard": lambda: __Lizard(),
        "pannuke": lambda: __PanNuke(),
        "tcga": lambda: __TCGA(),
    }
    if name.lower() in name_dict:
        return name_dict[name]()
    else:
        assert False, "Unknown dataset `%s`" % name
