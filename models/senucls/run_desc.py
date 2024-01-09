import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import cv2
from misc.utils import center_pad_to_shape, cropping_center
from .utils import crop_to_shape, dice_loss, mse_loss, xentropy_loss, get_bboxes,focal_loss,add_class,get_infer_bboxes
import os
from collections import OrderedDict

import gc
from .loss_functions import asym_unified_focal_loss
from sklearn.metrics import f1_score
####
# training_weights = [2.184124994534404, 5.38417317883769, 2.805669191091806] 
# Weights for each class [others,healthy,malignant] counting cells in crops

def empty_trash():
    gc.collect()
    torch.cuda.empty_cache()

def get_expanded_positions(original_positions, added_length):
    return [a*added_length + i for a in original_positions for i in range(added_length)]

def get_train_step(training_weights, only_epithelial):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def train_step(batch_data, run_info):
        # TODO: synchronize the attach protocol
        run_info, state_info = run_info
        loss_func_dict = {
            "bce": xentropy_loss,
            "dice": dice_loss,
            "mse": mse_loss,
            "focal": focal_loss,
            "new_seg_loss": asym_unified_focal_loss()
        }
        # use 'ema' to add for EMA calculation, must be scalar!
        result_dict = {"EMA": {}}
        track_value = lambda name, value: result_dict["EMA"].update({name: value})
        ####
        model = run_info["net"]["desc"]
        optimizer = run_info["net"]["optimizer"]

        ####
        imgs = batch_data["img"]
        # inst_map = batch_data["inst_map"]
        #cv2.imwrite('batch_img.png',np.array(imgs[0,:,:]))
        #cv2.imwrite('batch.png',np.uint8(np.array(inst_map[0,:,:])))
        #true_hv = batch_data["hv_map"]

        imgs = imgs.type(torch.float32)  # to NCHW
        imgs = imgs.permute(0, 3, 1, 2).contiguous()

        # HWC
        #inst_map = inst_map.to("cuda").type(torch.int64)
        #true_hv = true_hv.to("cuda").type(torch.float32)

        #batch_bboxes = torch.stack(batch_bboxes,0)
        #print(batch_bboxes.shape)
        # true_np_onehot = (F.one_hot(true_np, num_classes=2)).type(torch.float32)
        true_dict = {
            #"np": true_np_onehot,
            #"hv": true_hv,
        }

        if model.module.nr_types is not None:
            true_tp = batch_data["tp_map"]
            #print(true_tp.shape)

            # if not only_epithelial:
            #     true_tp += batch_data["np_map"]

            true_tp = torch.squeeze(true_tp).to(device).type(torch.int64)
            true_tp_onehot = F.one_hot(true_tp, num_classes=model.module.nr_types)
            true_tp_onehot = true_tp_onehot.type(torch.float32)
            true_dict["tp"] = true_tp_onehot
        #print(inst_map.shape)
        # type_map = batch_data["tp_map"]
        batch_boxes = batch_data['batch_boxes']
        batch_centers = batch_data['batch_centers']
        batch_edge_points = batch_data['batch_edge_points']
        batch_edge_indexs = batch_data['batch_edge_indexs']
        batch_pos_emb = batch_data['batch_pos_emb']
        # batch_length = batch_data['batch_length']
        batch_select_shape_feats = batch_data['batch_select_shape_feats']
        inst_classes = batch_data['inst_classes']


        inst_classes = torch.cat(inst_classes,0) - 1 # for one-hot encoding num of cell types (2/3) needs to be bigger than the values of the tensor (0,1/0,1,2)
        #print(inst_classes.shape)
        #print('count: ',torch.bincount(torch.squeeze(inst_classes).int()))
        true_inst_classes = torch.squeeze(inst_classes).to(device).type(torch.int64)

        true_inst_classes_onehot = F.one_hot(true_inst_classes, num_classes=model.module.nr_types-1)
        true_inst_classes_onehot = true_inst_classes_onehot.type(torch.float32)
        #print(true_inst_classes_onehot.shape)
        ####
        model.train()
        model.zero_grad()  # not rnn so not accumulate

        pred_dict = model(imgs,batch_boxes,batch_centers,batch_edge_points,batch_edge_indexs,batch_pos_emb,batch_select_shape_feats,mode='train')
        #pred_dict = OrderedDict(
            #[[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
        #)
        #pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)
        pred_map, pred_classes = pred_dict["tp"][0],pred_dict["tp"][1]
        pred_map = pred_map.permute(0, 2, 3, 1).contiguous()
        pred_map = F.softmax(pred_map, dim=-1)
        pred_classes = F.softmax(pred_classes, dim=-1)
        #print(pred_map.shape,pred_classes.shape)
        #print(true_tp_onehot.shape,true_inst_classes_onehot.shape)
        ####
        loss = 0
        loss_opts = run_info["net"]["extra_info"]["loss"]
        for branch_name in pred_dict.keys():
            for loss_name, loss_weight in loss_opts[branch_name].items():
                #print(loss_name)
                loss_func = loss_func_dict[loss_name]
                loss_args = [true_dict[branch_name], pred_map]
                if loss_name == "focal":
                    loss_args = [pred_classes, true_inst_classes_onehot, training_weights]
                term_loss = loss_func(*loss_args)
                track_value("loss_%s_%s" % (branch_name, loss_name), term_loss.cpu().item())
                #print(term_loss)
                loss += loss_weight * term_loss
        loss.backward()
        optimizer.step()
        track_value("overall_loss", loss.cpu().item())      
        ####
        del imgs, true_dict, pred_dict
        empty_trash()
        return result_dict
    
    return train_step

####
def get_valid_step(only_epithelial):
    def valid_step(batch_data, run_info):
        run_info, state_info = run_info
        ####
        model = run_info["net"]["desc"]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.eval()  # infer mode

        ####
        #true_np = batch_data["np_map"]
        #true_hv = batch_data["hv_map"]
        # inst_map = batch_data["inst_map"]
        imgs = batch_data["img"].type(torch.float32)  # to NCHW
        imgs= imgs.permute(0, 3, 1, 2).contiguous()

        # HWC

        true_dict = {
        }
        if model.module.nr_types is not None:
            true_tp = batch_data["tp_map"]

            # if not only_epithelial:
            #     true_tp += batch_data["np_map"]
            true_tp = torch.squeeze(true_tp).to(device).type(torch.int64)
            true_dict["tp"] = true_tp

        type_map = batch_data["tp_map"]
        batch_boxes = batch_data['batch_boxes']
        batch_centers = batch_data['batch_centers']
        batch_edge_points = batch_data['batch_edge_points']
        batch_edge_indexs = batch_data['batch_edge_indexs']
        batch_pos_emb = batch_data['batch_pos_emb']
        # batch_length = batch_data['batch_length']
        batch_select_shape_feats = batch_data['batch_select_shape_feats']
        inst_classes = batch_data['inst_classes']
        
        inst_classes =  torch.cat(inst_classes,0) - 1
        # --------------------------------------------------------------
        with torch.no_grad():  # dont compute gradient
            pred_dict = model(imgs,batch_boxes,batch_centers,batch_edge_points,batch_edge_indexs,batch_pos_emb,batch_select_shape_feats,mode='train')
            pred_map, pred_classes = pred_dict["tp"][0],pred_dict["tp"][1]
            pred_map = pred_map.permute(0, 2, 3, 1).contiguous()
            pred_classes = torch.argmax(pred_classes, dim=-1)
            type_map = torch.argmax(pred_map, dim=-1, keepdim=False)
            type_map = type_map.type(torch.float32)
            pred_dict["tp"] = type_map


        # * Its up to user to define the protocol to process the raw output per step!
        result_dict = {  # protocol for contents exchange within `raw`
            "raw": {
                "imgs": batch_data["img"].numpy(),
                #"pred_classes": pred_dict["np"].cpu().numpy(),
                #"pred_hv": pred_dict["hv"].cpu().numpy(),
            }
        }
        #print(true_dict["tp"].cpu().numpy().shape)
        #print(type_map.cpu().numpy().shape)
        if model.module.nr_types is not None:
            result_dict["raw"]["true_tp"] = true_dict["tp"].cpu().unsqueeze(-1).numpy()
            result_dict["raw"]["pred_tp"] = type_map.cpu().unsqueeze(-1).numpy()
            result_dict["raw"]["true_class"] = inst_classes.view_as(pred_classes).cpu().numpy()
            result_dict["raw"]["pred_class"] = pred_classes.cpu().numpy()
        empty_trash()
        return result_dict
    return valid_step

####

def infer_step(batch_data, model, mode='test'):

    ####
    patch_imgs = batch_data['img']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inst_map = batch_data["inst_map"]
    patch_imgs_gpu = patch_imgs.to(device).type(torch.float32)  # to NCHW
    patch_imgs_gpu = patch_imgs_gpu.permute(0, 3, 1, 2).contiguous()
    edge_num = 4
    point_num = 18

    ####
    model.eval()  # infer mode
    outputs = []
    # --------------------------------------------------------------
    with torch.no_grad():  # dont compute gradient
        for i in range(inst_map.shape[0]):
            single_map = inst_map[i,:,:].reshape(inst_map.shape[1],inst_map.shape[2])
            #print(single_map.shape)
            if single_map.max() > 0:
                true_bboxes,centers,edge_points,edge_index,pos_emd,select_shape_feats = get_infer_bboxes(single_map,edge_num,point_num)
                pred_dict = model(patch_imgs_gpu[i].unsqueeze(0),[true_bboxes],[centers],[edge_points],[edge_index],[pos_emd],[select_shape_feats],mode='test')
                pred_map, pred_classes = pred_dict["tp"][0],pred_dict["tp"][1]
                pred_map = pred_map.permute(0, 2, 3, 1).contiguous()
                #pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:]
                pred_map = F.softmax(pred_map, dim=-1)
                pred_classes = F.softmax(pred_classes, dim=-1)
    
                prediction = torch.argmax(pred_classes, 1).cpu().numpy()
                #print(np.unique(prediction) + 1)
                output = add_class(single_map,prediction)
                #print(np.unique(output))
                outputs.append(output)
            else:
                output = np.zeros((single_map.shape[0],single_map.shape[1]))
                outputs.append(output)
    #outputs = np.stack(outputs)
    outputs = np.stack(outputs)
    #print(output.shape)
    return outputs


####
def viz_step_output(raw_data, nr_types=None):
    """
    `raw_data` will be implicitly provided in the similar format as the 
    return dict from train/valid step, but may have been accumulated across N running step
    """

    imgs = raw_data["img"]
    #true_np, pred_np = raw_data["np"]
    #true_hv, pred_hv = raw_data["hv"]

    true_tp, pred_tp = raw_data["tp"]

    aligned_shape = [list(imgs.shape), list(true_tp.shape), list(pred_tp.shape)]
    aligned_shape = np.min(np.array(aligned_shape), axis=0)[1:3]

    cmap = plt.get_cmap("jet")

    def colorize(ch, vmin, vmax):
        """
        Will clamp value value outside the provided range to vmax and vmin
        """
        ch = np.squeeze(ch.astype("float32"))
        ch[ch > vmax] = vmax  # clamp value
        ch[ch < vmin] = vmin
        ch = (ch - vmin) / (vmax - vmin + 1.0e-16)
        # take RGB from RGBA heat map
        ch_cmap = (cmap(ch)[..., :3] * 255).astype("uint8")
        # ch_cmap = center_pad_to_shape(ch_cmap, aligned_shape)
        return ch_cmap

    viz_list = []
    for idx in range(imgs.shape[0]):
        # img = center_pad_to_shape(imgs[idx], aligned_shape)
        img = cropping_center(imgs[idx], aligned_shape)

        true_viz_list = [img]
        # cmap may randomly fails if of other types
        if nr_types is not None:  # TODO: a way to pass through external info
            true_viz_list.append(colorize(true_tp[idx], 0, nr_types))
        true_viz_list = np.concatenate(true_viz_list, axis=1)

        pred_viz_list = [img]

        if nr_types is not None:
            pred_viz_list.append(colorize(pred_tp[idx], 0, nr_types))
        pred_viz_list = np.concatenate(pred_viz_list, axis=1)

        viz_list.append(np.concatenate([true_viz_list, pred_viz_list], axis=0))
    viz_list = np.concatenate(viz_list, axis=0)
    return viz_list


####
from itertools import chain


def proc_valid_step_output(raw_data, nr_types=None):
    # TODO: add auto populate from main state track list
    track_dict = {"scalar": {}, "image": {}}

    def track_value(name, value, vtype):
        return track_dict[vtype].update({name: value})

    def _dice_info(true, pred, label):
        true = np.array(true == label, np.int32)
        pred = np.array(pred == label, np.int32)
        inter = (pred * true).sum()
        total = (pred + true).sum()
        return inter, total

    over_inter = 0
    over_total = 0
    over_correct = 0

    # * TP statistic
    pred_tp = raw_data["pred_tp"]
    true_tp = raw_data["true_tp"]
    pred_class = F.one_hot(torch.tensor(raw_data["pred_class"]).long()).numpy()
    true_class = F.one_hot(torch.tensor(raw_data["true_class"]).long()).numpy()
    f1s = []
    #print(len(true_tp))
    #print(pred_tp.shape)
    for type_id in range(0, nr_types):
        over_inter = 0
        over_total = 0
        for idx in range(len(raw_data["true_tp"])):
            patch_pred_tp = pred_tp[idx]
            patch_true_tp = true_tp[idx]
            inter, total = _dice_info(patch_true_tp, patch_pred_tp, type_id)
            over_inter += inter
            over_total += total
        dice_tp = 2 * over_inter / (over_total + 1.0e-8)
        track_value(f"tp_dice_{type_id}", dice_tp, "scalar")
        if type_id > 0:
            f1 = f1_score(true_class[:,type_id-1], pred_class[:,type_id-1])
            f1s.append(f1)
            track_value(f"tp_f1-score_{type_id}", f1, "scalar")
    
    # f1 = f1_score(true_class, pred_class, average='weighted')
    true_class_n = true_class.sum(0)[-2:]
    f1w = f1s[-2:]
    f1w = sum([f1w[i]*true_class_n[i]/true_class_n.sum() for i in range(len(f1w))])

    track_value(f"tp_f1-score_w", f1w, "scalar")


    # *
    imgs = raw_data["imgs"]
    selected_idx = np.random.randint(0, len(imgs), size=(8,)).tolist()
    imgs = np.array([imgs[idx] for idx in selected_idx])
    viz_raw_data = {"img": imgs}


    true_tp = np.array([true_tp[idx] for idx in selected_idx])
    pred_tp = np.array([pred_tp[idx] for idx in selected_idx])
    viz_raw_data["tp"] = (true_tp, pred_tp)
    #print('1111111',viz_raw_data)
    viz_fig = viz_step_output(viz_raw_data, nr_types)
    track_dict["image"]["output"] = viz_fig

    return track_dict
