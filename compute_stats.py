import argparse
import cProfile as profile
import glob
import os

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
from natsort import natsorted

from metrics.stats_utils import (
    get_dice_1,
    get_fast_aji,
    get_fast_aji_plus,
    get_fast_dice_2,
    get_fast_pq,
    remap_label,
    pair_coordinates
)
from misc.utils import parse_json_file

def filter_by_true_files(file_list:list, true_dict_list:list):
    true_files = [tile['img_id'] for tile in true_dict_list]
    file_list = [f for f in file_list if os.path.splitext(os.path.basename(f))[0] in true_files]
    return file_list

def include_all_cells(pred_dir:str, pred_dir_before_refine:str):

    pred_info = sio.loadmat(pred_dir)
    pred_info_before_refine = sio.loadmat(pred_dir_before_refine)
    
    centroids_after = pred_info['inst_centroid']
    centroids_before = pred_info_before_refine['inst_centroid']
    inst_type_after = pred_info['inst_type'].flatten()
    inst_type_before = pred_info_before_refine['inst_type'].flatten()

    if 'PGT' in pred_dir_before_refine:
            inst_type_before -= 1

    # centroids are arrays of shape (N, 2). I want to find for every centroid_after (i, 2), the centroid_before (j, 2) that is closest to it.
    # I then want to replace the inst_type_before[j] with inst_type_after[i]

    for i, centroid_after in enumerate(centroids_after):
        centroid_after = centroid_after.reshape((1, 2))
        distances = np.linalg.norm(centroids_before - centroid_after, axis=1)
        j = np.argmin(distances)
        inst_type_before[j] = inst_type_after[i]

    pred_info_before_refine['inst_type'] = inst_type_before
    return pred_info_before_refine



def run_nuclei_type_stat(pred_dir, pred_dir_before_refine, true_info, root_dir, merge_epit, type_uid_list=None, exhaustive=True, only_epit=True,):
    """GT must be exhaustively annotated for instance location (detection).

    Args:
        true_dir, pred_dir: Directory contains .mat annotation for each image. 
                            Each .mat must contain:
                    --`inst_centroid`: Nx2, contains N instance centroid
                                       of mass coordinates (X, Y)
                    --`inst_type`    : Nx1: type of each instance at each index
                    `inst_centroid` and `inst_type` must be aligned and each
                    index must be associated to the same instance
        type_uid_list : list of id for nuclei type which the score should be calculated.
                        Default to `None` means available nuclei type in GT.
        exhaustive : Flag to indicate whether GT is exhaustively labelled
                     for instance types
                     
    """
    print("Predictions in dir: ", pred_dir)
    file_list = natsorted([f.path for f in os.scandir(pred_dir) if f.is_file()])
    file_list_before_refine = natsorted([f.path for f in os.scandir(pred_dir_before_refine) if f.is_file()])

    paired_all = []  # unique matched index pair
    unpaired_true_all = []  # the index must exist in `true_inst_type_all` and unique
    unpaired_pred_all = []  # the index must exist in `pred_inst_type_all` and unique
    true_inst_type_all = []  # each index is 1 independent data point
    pred_inst_type_all = []  # each index is 1 independent data point

    true_dict_list = parse_json_file(true_info)
    true_dir = os.path.join(args.root_dir, os.path.dirname(true_dict_list[0]['mat_file']))
    file_list = filter_by_true_files(file_list, true_dict_list)
    file_list_before_refine = filter_by_true_files(file_list_before_refine, true_dict_list)
    
    file_list = [f for f in file_list if 'TMA' not in f]
    file_list_before_refine = [f for f in file_list_before_refine if 'TMA' not in f]
    for file_idx, filename in enumerate(file_list[:]):

        filename = os.path.basename(filename)
        filename_before_refine = os.path.basename(file_list_before_refine[file_idx])
        basename = os.path.splitext(filename)[0]
        basename_before_refine = os.path.splitext(filename_before_refine)[0]
        dict_list_pos = np.where([basename == tile['img_id'] for tile in true_dict_list])[0][0]
        
        is_endo = True if 'endo' in true_dir.lower() else False
        is_tcga = True if 'tcga' in basename.lower() else False
        centroid_key = 'centroid' if not is_tcga and not is_endo else 'inst_centroid'
        type_key = 'class' if not is_tcga and not is_endo else 'inst_type'
        
        basename_true = basename[:-2] if is_endo else basename
        true_info = sio.loadmat(os.path.join(true_dir, basename_true + ".mat"))
        # dont squeeze, may be 1 instance exist
        true_centroid = (true_info[centroid_key]).astype("float32")
        true_inst_type = (true_info[type_key]).astype("int32")

        if only_epit and not is_tcga and not is_endo: 
            epit_type = true_dict_list[dict_list_pos]['malignant'] + 1
            true_inst_type[true_inst_type != 2] = 0
            true_inst_type[true_inst_type == 2] = epit_type
            
        if true_centroid.shape[0] != 0:
            # true_inst_type = true_inst_type[:, 0]
            true_inst_type = true_inst_type.flatten()
        else:  # no instance at all
            true_centroid = np.array([[0, 0]])
            true_inst_type = np.array([0])


        # pred_info = sio.loadmat(os.path.join(pred_dir, basename + ".mat"))
        # pred_info_before_refine = sio.loadmat(os.path.join(pred_dir_before_refine, basename_before_refine + ".mat"))
        pred_info = include_all_cells(os.path.join(pred_dir, basename + ".mat"), os.path.join(pred_dir_before_refine, basename_before_refine + ".mat"))
        # dont squeeze, may be 1 instance exist
        pred_centroid = (pred_info["inst_centroid"]).astype("float32") if 'inst_centroid' in pred_info.keys() else (pred_info["inst_centroids"]).astype("float32")
        pred_inst_type = (pred_info["inst_type"]).astype("int32")
        

        if pred_centroid.shape[0] != 0:
            # pred_inst_type = pred_inst_type[:, 0]
            pred_inst_type = pred_inst_type.flatten()
        else:  # no instance at all
            pred_centroid = np.array([[0, 0]])
            pred_inst_type = np.array([0])

        # ! if take longer than 1min for 1000 vs 1000 pairing, sthg is wrong with coord
        paired, unpaired_true, unpaired_pred = pair_coordinates(
            true_centroid, pred_centroid, 6
        )

        # * Aggreate information
        # get the offset as each index represent 1 independent instance
        true_idx_offset = (
            true_idx_offset + true_inst_type_all[-1].shape[0] if file_idx != 0 else 0
        )
        pred_idx_offset = (
            pred_idx_offset + pred_inst_type_all[-1].shape[0] if file_idx != 0 else 0
        )
        true_inst_type_all.append(true_inst_type)
        pred_inst_type_all.append(pred_inst_type)

        # increment the pairing index statistic
        if paired.shape[0] != 0:  # ! sanity
            paired[:, 0] += true_idx_offset
            paired[:, 1] += pred_idx_offset
            paired_all.append(paired)

        unpaired_true += true_idx_offset
        unpaired_pred += pred_idx_offset
        unpaired_true_all.append(unpaired_true)
        unpaired_pred_all.append(unpaired_pred)

    paired_all = np.concatenate(paired_all, axis=0)
    unpaired_true_all = np.concatenate(unpaired_true_all, axis=0)
    unpaired_pred_all = np.concatenate(unpaired_pred_all, axis=0)
    true_inst_type_all = np.concatenate(true_inst_type_all, axis=0)
    pred_inst_type_all = np.concatenate(pred_inst_type_all, axis=0)

    paired_true_type = true_inst_type_all[paired_all[:, 0]]
    paired_pred_type = pred_inst_type_all[paired_all[:, 1]]
    unpaired_true_type = true_inst_type_all[unpaired_true_all]
    unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]

    ###
    def _w_f1_type(paired_true, paired_pred, unpaired_true=None, unpaired_pred=None, type_id=None, simple=True):
        # For this computation we do not care about cells without cell type. Thus we delete all cells
        # with paired_true==0 or unpaired_true==0 from the arrays.

        epit_pos_paired = paired_true != 0
        paired_true = paired_true[epit_pos_paired]
        paired_pred = paired_pred[epit_pos_paired]

        epit_pos_unpaired = unpaired_true != 0
        unpaired_true = unpaired_true[epit_pos_unpaired]

        epit_pos_unpaired = unpaired_true != 0
        unpaired_true = unpaired_true[epit_pos_unpaired]
        epit_pos_unpaired = unpaired_pred != 0
        unpaired_pred = unpaired_pred[epit_pos_unpaired]

        # if simple:
            # from sklearn.metrics import f1_score
            # return f1_score(paired_true, paired_pred, average='weighted', labels=[1,2])
        # else:

        f1s = {}
        accuracies = {}
        precisions = {}
        recalls = {}
        ratios = {}
        total_count = len(paired_true) if simple else len(paired_true) + len(unpaired_true)  
        for label in np.unique(paired_true):
            tp = ((paired_true == label) & (paired_pred == label)).sum()
            tn = ((paired_true != label) & (paired_pred != label)).sum()
            fp = ((paired_true != label) & (paired_pred == label)).sum()
            fn = ((paired_true == label) & (paired_pred != label)).sum()

            total_count_type = (paired_true == label).sum()

            # unpaired_pred are false cells, and unpaired_true are missed cells
            if not simple:
                fp += (unpaired_pred == label).sum()
                fn += (unpaired_true == label).sum()
                total_count_type += (unpaired_true == label).sum()
            f1s[label] = (2 * tp) / (2 * tp + fp + fn)
            accuracies[label] = (tp + tn) / (tp + tn + fp + fn)
            precisions[label] = tp / (tp + fp)
            recalls[label] = tp / (tp + fn)
            ratios[label] = total_count_type / total_count

        f1s['weighted'] = sum([f1s[label] * ratios[label] for label in f1s.keys()])
        accuracies['weighted'] = sum([accuracies[label] * ratios[label] for label in accuracies.keys()])
        precisions['weighted'] = sum([precisions[label] * ratios[label] for label in precisions.keys()])
        recalls['weighted'] = sum([recalls[label] * ratios[label] for label in recalls.keys()])

        # print(f"Stats counting missed cells: {not simple}")
        # print(f"\nw-Accuracy: \n {accuracies}")
        # print(f"\nw-Precision: \n {precisions}")
        # print(f"\nw-Recall: \n {recalls}")
        # print(f"\nw-F1: \n {f1s}")

        metrics = dict(
            Accuracy = accuracies,
            Precision = precisions,
            Recall = recalls,
            F1 = f1s,)

        return metrics            


    def _f1_type(paired_true, paired_pred, unpaired_true, unpaired_pred, type_id, w):
        type_samples = (paired_true == type_id) | (paired_pred == type_id)

        paired_true = paired_true[type_samples]
        paired_pred = paired_pred[type_samples]

        tp_dt = ((paired_true == type_id) & (paired_pred == type_id)).sum()
        tn_dt = ((paired_true != type_id) & (paired_pred != type_id)).sum()
        fp_dt = ((paired_true != type_id) & (paired_pred == type_id)).sum()
        fn_dt = ((paired_true == type_id) & (paired_pred != type_id)).sum()

        if not exhaustive:
            ignore = (paired_true == -1).sum()
            fp_dt -= ignore

        fp_d = (unpaired_pred == type_id).sum()
        fn_d = (unpaired_true == type_id).sum()

        f1_type = (2 * (tp_dt + tn_dt)) / (
            2 * (tp_dt + tn_dt)
            + w[0] * fp_dt
            + w[1] * fn_dt
            + w[2] * fp_d
            + w[3] * fn_d
        )
        return f1_type

    # overall
    # * quite meaningless for not exhaustive annotated dataset
    # merge healthy and malignant classes (1,2) into class 1 for paired and unpaired types
    if merge_epit:
        paired_true_type[paired_true_type == 2] = 1
        paired_pred_type[paired_pred_type == 2] = 1
        unpaired_true_type[unpaired_true_type == 2] = 1
        unpaired_pred_type[unpaired_pred_type == 2] = 1
    metrics_simple = _w_f1_type(paired_true_type, paired_pred_type, unpaired_true_type, unpaired_pred_type, simple=True)
    metrics_full = _w_f1_type(paired_true_type, paired_pred_type, unpaired_true_type, unpaired_pred_type, simple=False)

    w = [1, 1]
    tp_d = paired_pred_type.shape[0]
    fp_d = unpaired_pred_type.shape[0] # Falsely detected cells
    fn_d = unpaired_true_type.shape[0] # Missed cells

    tp_tn_dt = (paired_pred_type == paired_true_type).sum()
    fp_fn_dt = (paired_pred_type != paired_true_type).sum()

    if not exhaustive:
        ignore = (paired_true_type == -1).sum()
        fp_fn_dt -= ignore

    acc_type = tp_tn_dt / (tp_tn_dt + fp_fn_dt)
    f1_d = 2 * tp_d / (2 * tp_d + w[0] * fp_d + w[1] * fn_d)

    w = [2, 2, 1, 1]

    if type_uid_list is None:
        type_uid_list = np.unique(true_inst_type_all).tolist()

    results_list = [f1_d, acc_type]
    for type_uid in type_uid_list:
        f1_type = _f1_type(
            paired_true_type,
            paired_pred_type,
            unpaired_true_type,
            unpaired_pred_type,
            type_uid,
            w,
        )
        results_list.append(f1_type)

    np.set_printoptions(formatter={"float": "{: 0.5f}".format})
    print(np.array(results_list))
    metrics_simple['F1d'] = dict(all=f1_d)
    metrics_full['F1d'] = dict(all=f1_d)
    return metrics_simple, metrics_full


def run_nuclei_inst_stat(pred_dir, true_dir, root_dir, print_img_stats=False, ext=".mat"):
    # print stats of each image
    print(pred_dir)

    file_list = glob.glob("%s/*%s" % (pred_dir, ext))
    file_list.sort()  # ensure same order
    file_list_true = [os.path.join(root_dir, tile['mat_file']) for tile in parse_json_file(true_dir)]
    metrics = [[], [], [], [], [], []]
    for filename in file_list[:]:
        filename = os.path.basename(filename)
        basename = filename.split(".")[0]

        true = sio.loadmat(os.path.join(true_dir, basename + ".mat"))
        true = (true["inst_map"]).astype("int32")

        pred = sio.loadmat(os.path.join(pred_dir, basename + ".mat"))
        pred = (pred["inst_map"]).astype("int32")

        # to ensure that the instance numbering is contiguous
        pred = remap_label(pred, by_size=False)
        true = remap_label(true, by_size=False)

        pq_info = get_fast_pq(true, pred, match_iou=0.5)[0]
        metrics[0].append(get_dice_1(true, pred))
        metrics[1].append(get_fast_aji(true, pred))
        metrics[2].append(pq_info[0])  # dq
        metrics[3].append(pq_info[1])  # sq
        metrics[4].append(pq_info[2])  # pq
        metrics[5].append(get_fast_aji_plus(true, pred))

        if print_img_stats:
            print(basename, end="\t")
            for scores in metrics:
                print("%f " % scores[-1], end="  ")
            print()
    ####
    metrics = np.array(metrics)
    metrics_avg = np.mean(metrics, axis=-1)
    np.set_printoptions(formatter={"float": "{: 0.5f}".format})
    print(metrics_avg)
    metrics_avg = list(metrics_avg)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        help="mode to run the measurement,"
        "`type` for nuclei instance type classification or"
        "`instance` for nuclei instance segmentation",
        nargs="?",
        default="instance",
        const="instance",
    )
    parser.add_argument(
        "--pred_dir", help="point to output dir", nargs="?", default="", const=""
    )
    parser.add_argument(
        "--pred_dir_before_refine", help="point to output dir", nargs="?", default="", const=""
    )
    parser.add_argument(
        "--true_dir", help="point to ground truth dir", nargs="?", default="", const=""
    )
    parser.add_argument(
        "--root_dir", help="point to root dir", nargs="?", default="", const=""
    )
    parser.add_argument('--cv', type=int, default=None, help='run cross validation (mean +- std)')
    parser.add_argument('--merge_epit', action='store_true', help='merge healthy and malignant classes (1,2) into class 1 for paired and unpaired types')
    args = parser.parse_args()

    

    if args.mode == "instance":
        run_nuclei_inst_stat(args.pred_dir, args.true_dir, args.root_dir, print_img_stats=False)
    if args.mode == "type":
        if args.cv is None:
            simple, full = run_nuclei_type_stat(args.pred_dir, args.pred_dir_before_refine, args.true_dir,  args.root_dir, args.merge_epit)
        else:
            current_fold = args.pred_dir.split('/cv')[1][0]
            current_fold_before_refine = args.pred_dir_before_refine.split('/cv')[1][0]
            true_dir = args.true_dir
            metrics_cv_simple = {}
            metrics_cv_full = {}
            for i in range(0, args.cv):

                # Step 1: start from fold 0 by changing the passed fold from args.pred_dir to fold_i
                pred_dir = args.pred_dir.replace(f'cv{current_fold}', f'cv{i}')
                pred_dir_before_refine = args.pred_dir_before_refine.replace(f'cv{current_fold_before_refine}', f'cv{i}')
                if 'cv' in true_dir:
                    true_dir = args.true_dir.replace(f'cv{current_fold}', f'cv{i}')
                
                
                simple, full = run_nuclei_type_stat(pred_dir, pred_dir_before_refine, true_dir,  args.root_dir, args.merge_epit)
                # The metrics are organized in a dictionary with the following keys:
                #   F1: dict, with keys being the class (and a weighted key) and values being the F1 scores
                #   Accuracy: dict, with keys being the class (and a weighted key) and values being the Accuracy scores
                #   Precision: dict, with keys being the class (and a weighted key) and values being the Precision scores
                #   Recall: dict, with keys being the class (and a weighted key) and values being the Recall scores

                # The goal is to have one dict with one dict per metric as keys (where keys are the class and a weighted key) and the values are a list of scores across the folds

                # Step 2: append the metrics to the metrics_cv dict
                for metric in simple.keys():
                    if metric not in metrics_cv_simple.keys():
                        metrics_cv_simple[metric] = {}
                        metrics_cv_full[metric] = {}
                    for key in simple[metric].keys():
                        if key not in metrics_cv_simple[metric].keys():
                            metrics_cv_simple[metric][key] = []
                            metrics_cv_full[metric][key] = []
                        metrics_cv_simple[metric][key].append(simple[metric][key])
                        metrics_cv_full[metric][key].append(full[metric][key])
            # Step 3: compute mean and std for each metric and class in metrics_cv and print:
            print('Simple metrics:')
            for metric in metrics_cv_simple.keys():
                print(f"{metric}")
                for key in metrics_cv_simple[metric].keys():
                    print(f"\t{key}: {100*np.mean(metrics_cv_simple[metric][key]):.2f} \u00B1 {100*np.std(metrics_cv_simple[metric][key]):.2f}")
            print('Full metrics:')
            for metric in metrics_cv_full.keys():
                print(f"\n{metric}")
                for key in metrics_cv_full[metric].keys():
                    print(f"\t{key}: {100*np.mean(metrics_cv_full[metric][key]):.2f} \u00B1 {100*np.std(metrics_cv_full[metric][key]):.2f}")
                            
