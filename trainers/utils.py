from tkinter.tix import Tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import os.path as osp
import json
import torch
from sklearn.metrics import precision_recall_curve
from collections import OrderedDict
import random
import time
import pdb
import math
from collections import OrderedDict

def plotLogitsMap(outputs, label, save_path, fig_title, max_lines=1000):
    fig, ax = plt.subplots(figsize=(5, 200))
    Softmax = torch.nn.Softmax(dim=1)
    output_m = Softmax(outputs)
    output_m = outputs.cpu().detach().numpy()

    pred = outputs.max(1)[1]
    matches = pred.eq(label).float()
    output_m = np.sort(output_m)
    output_m = output_m[:,::-1]
    output_m = output_m[:,:5]
    output_m_index = output_m[:,0].argsort()
    output_m = output_m[output_m_index]
    output_m = output_m[::-1,:]
    matches = matches[output_m_index]
    matches = torch.flip(matches, dims=[0])
    matches = matches.cpu().detach().numpy()


    if len(matches) > max_lines:
        gap = int(len(matches) / 1000)
        index = np.arange(0, gap*1000, gap, int)
        output_m = output_m[index]
        matches = matches[index]
    print(save_path)
    matches = matches.tolist()


    im = ax.imshow(output_m, aspect='auto')
    ax.set_yticks(np.arange(output_m.shape[0]), labels=matches)
    for i, label in enumerate(ax.get_yticklabels()):
        if (int(matches[i])==0):
            label.set_color('red')
        elif (int(matches[i])==1):
            label.set_color('green')

    for i in range(output_m.shape[0]):
        for j in range(output_m.shape[1]):
            text = ax.text(j, i, str(round(output_m[i, j],2)),
                        ha="center", va="center", color="w")
    plt.title(fig_title)
    plt.savefig(save_path)
    plt.close()

def plotPRMap(outputs, label, save_path, fig_title):
    plt.figure(figsize=(15,15))
    plt.title('{} Precision/Recall Curve'.format(fig_title))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    output_m = outputs.cpu().detach().numpy()
    pred = outputs.max(1)[1]
    matches = pred.eq(label).float()
    output_m = np.sort(output_m)
    output_m = output_m[:,::-1]
    output_m = output_m[:,:5]
    output_m_index = output_m[:,0].argsort()
    output_m = output_m[output_m_index]
    output_m = output_m[::-1,:]
    matches = matches[output_m_index]
    matches = torch.flip(matches, dims=[0])
    matches = matches.cpu().detach().numpy()
    precision, recall, thresholds = precision_recall_curve(matches, output_m[:,0])
    plt.plot(recall, precision)

    step = 0
    for a, b, text in zip(recall, precision, thresholds):
        # if float(text) % 0.05 == 0:
        if step % 40 == 0:
            plt.text(a, b, text, ha='center', va='bottom', fontsize=10, color='blue')
            plt.plot(a, b, marker='o', color='red')
        step += 1
    plt.grid(ls='--')
    plt.savefig(save_path)
    plt.close()

def select_top_k_similarity_per_class(outputs, img_paths, K=1, image_features=None, is_softmax=True):
    # print(outputs.shape)
    if is_softmax:
        outputs = torch.nn.Softmax(dim=1)(outputs)     # logits -> prob
    output_m = outputs.cpu().detach().numpy()
    output_ori = outputs.cpu().detach()
    output_m_max = output_m.max(axis=1)
    output_m_max_id = np.argsort(-output_m_max)
    output_m = output_m[output_m_max_id]
    img_paths = img_paths[output_m_max_id]
    output_m_max = output_m_max[output_m_max_id]
    output_ori = output_ori[output_m_max_id]
    ids = (-output_m).argsort()[:, 0]    # pred label

    if image_features is not None:
        image_features = image_features.cpu().detach()
        image_features = image_features[output_m_max_id]

    predict_label_dict = {}
    predict_conf_dict = {}
    from tqdm import tqdm
    for id in tqdm(list(set(ids.tolist()))):
        index = np.where(ids==id)
        conf_class = output_m_max[index]
        output_class = output_ori[index]
        img_paths_class = img_paths[index]

        if image_features is not None:
            img_features = image_features[index]
            if K >= 0:
                for img_path, img_feature, conf, logit in zip(img_paths_class[:K], img_features[:K], conf_class[:K], output_class):
                    if '/data/' in img_path:
                        img_path = './data/' + img_path.split('/data/')[1]
                    predict_label_dict[img_path] = [id, img_feature, conf, logit]
            else:
                for img_path, img_feature, conf, logit in zip(img_paths_class, img_features, conf_class, output_class):
                    if '/data/' in img_path:
                        img_path = './data/' + img_path.split('/data/')[1]
                    predict_label_dict[img_path] = [id, img_feature, conf, logit]
        else:
            if K >= 0:
                for img_path, conf in zip(img_paths_class[:K], conf_class):
                    if '/data/' in img_path:
                        img_path = './data/' + img_path.split('/data/')[1]
                    predict_label_dict[img_path] = id
                    predict_conf_dict[img_path] = conf
            else:
                for img_path, conf in zip(img_paths_class, conf_class):
                    if '/data/' in img_path:
                        img_path = './data/' + img_path.split('/data/')[1]
                    predict_label_dict[img_path] = id
                    predict_conf_dict[img_path] = conf
    return predict_label_dict, predict_conf_dict

def select_top_k_similarity_per_class_with_noisy_label(img_paths, K=1, random_seed=1, gt_label_dict=None, num_fp=0):
    predict_label_dict = {}
    if gt_label_dict is not None:
        ids = gt_label_dict.values()
        num_class = len(set(ids))
        gt_class_label_dict = {}
        for indx in range(num_class):
            gt_class_label_dict[indx] = np.array([])
        for ip, gt_label in gt_label_dict.items():
            gt_class_label_dict[gt_label] = np.append(gt_class_label_dict[gt_label], np.array(ip))
        
        img_paths_dict = {k: v for v, k in enumerate(img_paths)}   # impath: index
        fp_ids_chosen = set()
        rng = np.random.default_rng(seed=random_seed)
        acc_rate_dict = {}
        from tqdm import tqdm
        # noisy lebels - split data into TP and FP sets
        tp_gt_all_img_index_dict = {}
        fp_gt_all_img_index_dict = {}
        fp_gt_all_img_index_list = []
        for id in tqdm(list(set(ids))):
            # noisy lebels - fix candidates for 16 shot samples
            split = int(math.ceil((len(gt_class_label_dict[id]) * (0.5))))
            # noisy lebels - fix candidates for 16 shot samples
            gt_class_img_index = []
            for img in list(gt_class_label_dict[id]):
                gt_class_img_index.append(img_paths_dict[img])
            # if num_fp == 0:
            #     tp_gt_all_img_index_dict[id] = gt_class_img_index[:]
            # else:
            tp_gt_all_img_index_dict[id] = gt_class_img_index[:split]
            fp_gt_all_img_index_dict[id] = gt_class_img_index[split:]
            fp_gt_all_img_index_list.extend(gt_class_img_index[split:])
        fp_gt_all_img_index_set = set(fp_gt_all_img_index_list)
        # noisy lebels - split data into TP and FP sets

        for id in tqdm(list(set(ids))):
            # noisy lebels - randomly draw FP samples with their indice
            gt_class_img_index = tp_gt_all_img_index_dict[id]
            fp_ids_set = fp_gt_all_img_index_set.difference(gt_class_img_index, fp_gt_all_img_index_dict[id], fp_ids_chosen)
            fp_ids = random.choices(list(fp_ids_set), k=num_fp)
            fp_ids_chosen.update(fp_ids)
            # noisy lebels - randomly draw FP samples with their indice
            img_paths_class = img_paths[gt_class_img_index]
            if K >= 0:
                if len(img_paths_class) < K:
                    is_replace=True
                else:
                    is_replace=False
                K_array = rng.choice(len(img_paths_class), size=K, replace=is_replace)
                img_paths_class = img_paths_class[K_array]
                # noisy lebels - dilute with FP samples
                for i in range(num_fp):
                    img_paths_class[i] = img_paths[fp_ids][i]
                # noisy lebels - - dilute with FP samples
                print('---',id)
                print(img_paths_class)
                total = 0
                correct = 0
                for img_path in (img_paths_class):
                    if '/data/' in img_path:
                        img_path = './data/' + img_path.split('/data/')[1]
                    predict_label_dict[img_path] = id
                    if gt_label_dict[img_path] == predict_label_dict[img_path]:
                        correct += 1
                    total += 1
                    acc_rate_dict[id] = 100.0*(correct/total)
            else:
                for img_path in (img_paths_class):
                    if '/data/' in img_path:
                        img_path = './data/' + img_path.split('/data/')[1]
                    predict_label_dict[img_path] = id
        for class_id in acc_rate_dict:
            print('* class: {}, Acc Rate {:.2f}%, Noise Rate {:.2f}%'.format(class_id, acc_rate_dict[class_id], 100 - acc_rate_dict[class_id]))
        print('* average: {:.2f}%'.format(sum(acc_rate_dict.values())/len(acc_rate_dict)))
    else:
        print('GT dict is missing')
        pdb.set_trace()
    return predict_label_dict


def select_by_conf(outputs, img_paths, K=1, conf_threshold=None, is_softmax=True):
    # print(outputs.shape)
    if is_softmax:
        outputs = torch.nn.Softmax(dim=1)(outputs)
    output_m = outputs.cpu().detach().numpy()
    output_ori = outputs.cpu().detach()
    output_m_max = output_m.max(axis=1)
    output_m_max_id = np.argsort(-output_m_max)
    output_m = output_m[output_m_max_id]
    img_paths = img_paths[output_m_max_id]
    output_m_max = output_m_max[output_m_max_id]
    output_ori = output_ori[output_m_max_id]
    ids = (-output_m).argsort()[:, 0]


    predict_label_dict = {}
    predict_conf_dict = {}
    from tqdm import tqdm
    for id in tqdm(list(set(ids.tolist()))):
        index = np.where(ids==id)
        conf_class = output_m_max[index]
        output_class = output_ori[index]
        img_paths_class = img_paths[index]

        for img_path, conf in zip(img_paths_class, conf_class):
            if conf > conf_threshold:
                predict_label_dict[img_path] = id
                predict_conf_dict[img_path] = conf
    return predict_label_dict, predict_conf_dict

def select_top_k_similarity(outputs, img_paths, K=1, image_features=None, repeat=False):
    outputs = torch.nn.Softmax(dim=1)(outputs)
    output_m = outputs.cpu().detach().numpy()
    output_ori = outputs.cpu().detach()
    output_m_max = output_m.max(axis=1)
    output_m_max_id = np.argsort(-output_m_max)
    output_m = output_m[output_m_max_id]
    img_paths = img_paths[output_m_max_id]
    output_ori = output_ori[output_m_max_id]
    conf_class = output_m_max[output_m_max_id]
    ids = (-output_m).argsort()[:, 0]

    if image_features is not None:
        image_features = image_features.cpu().detach()
        image_features = image_features[output_m_max_id]

    predict_label_dict = {}
    predict_conf_dict = {}
    if image_features is not None:
        img_features = image_features
        if K >= 0:
            for img_path, img_feature, conf, logit in zip(img_paths_class[:K], img_features[:K], conf_class[:K], output_class):
                predict_label_dict[img_path] = [id, img_feature, conf, logit]
        else:
            for img_path, img_feature, conf, logit in zip(img_paths_class, img_features, conf_class, output_class):
                predict_label_dict[img_path] = [id, img_feature, conf, logit]
    else:
        if K >= 0:
            for img_path, conf, id in zip(img_paths[:K], conf_class[:K], ids[:K]):
                predict_label_dict[img_path] = id
                predict_conf_dict[img_path] = conf
        else:
            for img_path, conf, id in zip(img_paths, conf_class, ids):
                predict_label_dict[img_path] = id
                predict_conf_dict[img_path] = conf
    return predict_label_dict, predict_conf_dict


def select_top_by_value(outputs, img_paths, conf_threshold=0.95, image_features=None, repeat=False):
    outputs = torch.nn.Softmax(dim=1)(outputs)
    output_m = outputs.cpu().detach().numpy()
    output_ori = outputs.cpu().detach()
    output_m_max = output_m.max(axis=1)
    output_m_max_id = np.argsort(-output_m_max)
    output_m = output_m[output_m_max_id]
    img_paths = img_paths[output_m_max_id]
    output_ori = output_ori[output_m_max_id]
    conf_class = output_m_max[output_m_max_id]
    ids = (-output_m).argsort()[:, 0]

    if image_features is not None:
        image_features = image_features.cpu().detach()
        image_features = image_features[output_m_max_id]

    predict_label_dict = {}
    predict_conf_dict = {}
    if image_features is not None:
        img_features = image_features
        for img_path, img_feature, conf, logit in zip(img_paths_class, img_features, conf_class, output_class):
            if conf > conf_threshold:
                predict_label_dict[img_path] = [id, img_feature, conf, logit]
    else:
        for img_path, id, conf in zip(img_paths, ids, conf_class):
            if conf > conf_threshold:
                predict_label_dict[img_path] = id
                predict_conf_dict[img_path] = conf
    return predict_label_dict, predict_conf_dict


def caculate_noise_rate(predict_label_dict, train_loader, trainer, sample_level=False):
    gt_label_dict = {}
    for batch_idx, batch in enumerate(train_loader):
        input, label, impath = trainer.parse_batch_test_with_impath(batch)
        for l, ip in zip(label, impath):
            if '/data/' in ip:
                ip = './data/' + ip.split('/data/')[1]
            gt_label_dict[ip] = l

    total = 0
    correct = 0
    for item in predict_label_dict:
        if '/data/' in item:
            item = './data/' + item.split('/data/')[1]
        if gt_label_dict[item] == predict_label_dict[item]:
            correct += 1
        total += 1
    print('Acc Rate {:.4f}'.format(correct/total))


def caculate_noise_rate_analyze(predict_label_dict, train_loader, trainer, sample_level=False):
    gt_label_dict = {}
    for batch_idx, batch in enumerate(train_loader):
        input, label, impath = trainer.parse_batch_test_with_impath(batch)
        for l, ip in zip(label, impath):
            ip = './data/' + ip.split('/data/')[1]
            gt_label_dict[ip] = l
    total = 0
    correct = 0
    for item in predict_label_dict:
        if gt_label_dict[item] == predict_label_dict[item][0]:
            correct += 1
            if sample_level is True:
                print(gt_label_dict[item], 1)
        total += 1
    print('Acc Rate {:.4f}'.format(correct/total))
    print('Moise Rate {:.4f}'.format(1-correct/total))


def save_outputs(train_loader, trainer, predict_label_dict, dataset_name, text_features, backbone_name=None, tag='', seed =''):
    # predict_label_dict[img_path] = [id, img_feature, conf, logit]
    backbone_name = backbone_name.replace('/', '-')
    gt_pred_label_dict = {}
    for batch_idx, batch in enumerate(train_loader):
        input, label, impath = trainer.parse_batch_test_with_impath(batch)
        for l, ip in zip(label, impath):
            l = l.item()
            ip = './data/' + ip.split('/data/')[1]
            if l not in gt_pred_label_dict:
                gt_pred_label_dict[l] = []
                pred_label = predict_label_dict[ip][0]
                pred_v_feature = predict_label_dict[ip][1]

                conf = predict_label_dict[ip][2]
                logits = predict_label_dict[ip][3]
                gt_pred_label_dict[l].append([ip, pred_label, pred_v_feature, conf, logits])
            else:
                pred_label = predict_label_dict[ip][0]
                pred_v_feature = predict_label_dict[ip][1]
                conf = predict_label_dict[ip][2]
                logits = predict_label_dict[ip][3]
                gt_pred_label_dict[l].append([ip, pred_label, pred_v_feature, conf, logits])

    idx = 0
    v_distance_dict = {}
    v_features = []
    logits_list = []
    # get the class center of image features
    for label in gt_pred_label_dict:
        avg_feature = None
        for item in gt_pred_label_dict[label]:
            impath, pred_label, pred_v_feature = item[0], item[1], item[2],
            if avg_feature is None:
                avg_feature = pred_v_feature.clone()
            else:
                avg_feature += pred_v_feature.clone()
        avg_feature /= len(gt_pred_label_dict[label]) # class center
        v_distance_dict_per_class = {}
        for item in gt_pred_label_dict[label]:
            impath, pred_label, pred_v_feature, conf, logits = item[0], item[1], item[2], item[3], item[4]
            v_features.append(pred_v_feature)
            logits_list.append(logits)
            v_dis = torch.dist(avg_feature, pred_v_feature, p=2)
            v_distance_dict_per_class[impath] = [idx, v_dis.item(), conf.item(), pred_label] # id, visual distance, confidence, predicted label
            idx += 1
        v_distance_dict[label] = v_distance_dict_per_class

    v_features = torch.vstack(v_features)
    logits_tensor = torch.vstack(logits_list)

    if not os.path.exists('./analyze_results/{}{}/'.format(backbone_name, tag)):
        os.makedirs('./analyze_results/{}{}/'.format(backbone_name, tag))

    torch.save(v_features, './analyze_results/{}{}/{}_v_feature.pt'.format(backbone_name, tag, dataset_name, seed))
    torch.save(text_features, './analyze_results/{}{}/{}_l_feature{}.pt'.format(backbone_name, tag, dataset_name, seed))
    torch.save(logits_tensor, './analyze_results/{}{}/{}_logits{}.pt'.format(backbone_name, tag, dataset_name, seed))


    with open("./analyze_results/{}{}/{}{}.json".format(backbone_name, tag, dataset_name, seed), "w") as outfile:
        json.dump(v_distance_dict, outfile)



def select_top_k_similarity_per_class_with_high_conf(outputs, img_paths, K=1, image_features=None, repeat=False):
    outputs = torch.nn.Softmax(dim=1)(outputs)
    output_m = outputs.cpu().detach().numpy()
    output_ori = outputs.cpu().detach()
    output_m_max = output_m.max(axis=1)
    output_m_max_id = np.argsort(-output_m_max)
    output_m = output_m[output_m_max_id]
    img_paths = img_paths[output_m_max_id]
    output_m_max = output_m_max[output_m_max_id]
    output_ori = output_ori[output_m_max_id]
    ids = (-output_m).argsort()[:, 0]


    class_avg_conf = {}
    for id in list(set(ids.tolist())):
        index = np.where(ids==id)
        conf_class = output_m_max[index]
        class_avg_conf[id] = conf_class.sum() / conf_class.size

    selected_ids = sorted(class_avg_conf.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)[:int(0.8*len(class_avg_conf))]
    remain_ids = sorted(class_avg_conf.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)[int(0.8*len(class_avg_conf)):]

    selected_ids = [id[0] for id in selected_ids]
    remain_ids = [id[0] for id in remain_ids]

    if image_features is not None:
        image_features = image_features.cpu().detach()
        image_features = image_features[output_m_max_id]

    predict_label_dict = {}
    predict_conf_dict = {}


    for id in selected_ids:
        index = np.where(ids==id)
        conf_class = output_m_max[index]
        output_class = output_ori[index]
        img_paths_class = img_paths[index]
        if image_features is not None:
            img_features = image_features[index]
            if K >= 0:
                for img_path, img_feature, conf, logit in zip(img_paths_class[:K], img_features[:K], conf_class[:K], output_class[:K]):
                    img_path = './data/' + img_path.split('/data/')[1]
                    predict_label_dict[img_path] = [id, img_feature, conf, logit]
            else:
                for img_path, img_feature, conf, logit in zip(img_paths_class, img_features, conf_class, output_class):
                    img_path = './data/' + img_path.split('/data/')[1]
                    predict_label_dict[img_path] = [id, img_feature, conf, logit]
        else:
            if K >= 0:
                for img_path, conf in zip(img_paths_class[:K], conf_class):
                    img_path = './data/' + img_path.split('/data/')[1]
                    predict_label_dict[img_path] = id
                    predict_conf_dict[img_path] = conf
            else:
                for img_path, conf in zip(img_paths_class, conf_class):
                    img_path = './data/' + img_path.split('/data/')[1]
                    predict_label_dict[img_path] = id
                    predict_conf_dict[img_path] = conf
    return predict_label_dict, predict_conf_dict, remain_ids, selected_ids


def select_top_k_similarity_per_class_with_low_conf(outputs, img_paths, conf_threshold, remain_ids, selected_ids, K=2):
    # print(outputs.shape)
    outputs = torch.nn.Softmax(dim=1)(outputs)
    remain_ids_list = remain_ids
    remain_ids = np.sort(np.array(remain_ids).astype(np.int))
    remain_logits = -100*torch.ones(outputs.shape).half().cuda()
    remain_logits[:, remain_ids] = outputs[:, remain_ids] * 5
    remain_logits = torch.nn.Softmax(dim=1)(remain_logits.float())
    outputs = remain_logits


    output_m = outputs.cpu().detach().numpy()
    output_ori = outputs.cpu().detach()
    output_m_max = output_m.max(axis=1)
    output_m_max_id = np.argsort(-output_m_max)
    output_m = output_m[output_m_max_id]
    img_paths = img_paths[output_m_max_id]
    output_m_max = output_m_max[output_m_max_id]
    output_ori = output_ori[output_m_max_id]
    ids = (-output_m).argsort()[:, 0]


    predict_label_dict = {}
    predict_conf_dict = {}
    no_sample_ids = []

    for id in remain_ids_list:
        # print(id)
        is_id_have_sample = False
        index = np.where(ids==id)
        conf_class = output_m_max[index]
        output_class = output_ori[index]
        img_paths_class = img_paths[index]

        if K >= 0:
            for img_path, conf in zip(img_paths_class[:K], conf_class[:K]):
                print(conf)
                if conf > 0.4:
                    predict_label_dict[img_path] = id
                    predict_conf_dict[img_path] = conf
                    is_id_have_sample = True
        else:
            for img_path, conf in zip(img_paths_class, conf_class):
                predict_label_dict[img_path] = id
                predict_conf_dict[img_path] = conf
        if is_id_have_sample is False:
            no_sample_ids.append(id)

    print(no_sample_ids)
    return predict_label_dict, predict_conf_dict, no_sample_ids

def select_top_k_similarity_per_class_no_smaple(outputs, img_paths, no_sample_ids, K=16):
    outputs = torch.nn.Softmax(dim=1)(outputs)
    output_m = outputs.cpu().detach().numpy()
    output_ori = outputs.cpu().detach()
    output_m_max = output_m.max(axis=1)
    output_m_max_id = np.argsort(-output_m_max)
    output_m = output_m[output_m_max_id]
    img_paths = img_paths[output_m_max_id]
    output_m_max = output_m_max[output_m_max_id]
    output_ori = output_ori[output_m_max_id]
    ids = (-output_m).argsort()[:, 0]


    predict_label_dict = {}
    predict_conf_dict = {}

    for id in no_sample_ids:
        print(id)
        index = np.where(ids==id)
        conf_class = output_m_max[index]
        output_class = output_ori[index]
        img_paths_class = img_paths[index]

        if K >= 0:
            for img_path, conf in zip(img_paths_class[:K], conf_class[:K]):
                predict_label_dict[img_path] = id
                predict_conf_dict[img_path] = conf
        else:
            for img_path, conf in zip(img_paths_class, conf_class):
                predict_label_dict[img_path] = id
                predict_conf_dict[img_path] = conf
    return predict_label_dict, predict_conf_dict

def save_checkpoint(
    state,
    save_dir,
    is_best=False,
    remove_module_from_keys=True,
    model_name=""
):
    r"""Save checkpoint.

    Args:
        state (dict): dictionary.
        save_dir (str): directory to save checkpoint.
        is_best (bool, optional): if True, this checkpoint will be copied and named
            ``model-best.pth.tar``. Default is False.
        remove_module_from_keys (bool, optional): whether to remove "module."
            from layer names. Default is True.
        model_name (str, optional): model name to save.
    """
    mkdir_if_missing(save_dir)

    if remove_module_from_keys:
        # remove 'module.' in state_dict's keys
        state_dict = state["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v
        state["state_dict"] = new_state_dict

    # save model
    epoch = state["epoch"]
    batch = state['batch']
   
    if not model_name:
        
        model_name = "model.pth.tar_" + str(epoch) + "_" + str(batch)
    # from IPython import embed
    # embed()
    fpath = osp.join(save_dir, model_name)
    torch.save(state, fpath)
    print(f"Checkpoint saved to {fpath}")

    # save current model name
    checkpoint_file = osp.join(save_dir, "checkpoint")
    checkpoint = open(checkpoint_file, "w+")
    checkpoint.write("{}\n".format(osp.basename(fpath)))
    checkpoint.close()

    if is_best:
        best_fpath = osp.join(osp.dirname(fpath), "model-best.pth.tar")
        shutil.copy(fpath, best_fpath)
        print('Best checkpoint saved to "{}"'.format(best_fpath))


def mkdir_if_missing(dirname):
    """Create dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def update_or_create_csv(dictionary, csv_file_path):
    """Save and update results to csv file."""
    if os.path.exists(csv_file_path):
        existing_df = pd.read_csv(csv_file_path)
        existing_data = existing_df.to_dict(orient="list")

        for key, value in dictionary.items():
            if key not in existing_data:
                existing_data[key] = value

        max_length = max(len(col) for col in existing_data.values())
        for key in existing_data:
            while len(existing_data[key]) < max_length:
                existing_data[key].append(None)

        updated_df = pd.DataFrame(existing_data)
        updated_df.to_csv(csv_file_path, index=False, encoding="utf-8")

    else:
        subdir = csv_file_path.rsplit('/', 1)[0]
        mkdir_if_missing(subdir)
        df = pd.DataFrame(dictionary)
        df.to_csv(csv_file_path, index=False, encoding="utf-8")

def update_or_create_csv_dict(dictionary, csv_file_path):
    """Save and update results to CSV file."""
    if os.path.exists(csv_file_path):
        # 读取现有 CSV 文件
        existing_df = pd.read_csv(csv_file_path)
        
        # 将 CSV 数据转换为字典格式
        existing_data = existing_df.to_dict(orient="list")

        # 更新现有数据字典：添加新键或扩展已有键
        for key, value in dictionary.items():
            if key not in existing_data:
                existing_data[key] = value
            else:
                existing_data[key].extend(value)

        # 计算合并后列的最大长度
        max_length = max(len(col) for col in existing_data.values())

        # 使用 Pandas 自动补齐较短的列
        aligned_data = {key: value + [None] * (max_length - len(value)) for key, value in existing_data.items()}
        updated_df = pd.DataFrame(aligned_data)

    else:
        # 如果文件不存在，直接创建新 DataFrame
        updated_df = pd.DataFrame(dictionary)

    # 保存到 CSV 文件
    updated_df.to_csv(csv_file_path, index=False, encoding="utf-8")