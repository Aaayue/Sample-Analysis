from pycocotools.coco import COCO
from sklearn.metrics import confusion_matrix
import numpy as np
import json
import time
from mmdet.apis import init_detector, inference_detector, show_result
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
import mmcv
import argparse
from tqdm import tqdm
import shutil
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def detect_image(model, img):
    result = inference_detector(model, img)
    labels = [
        np.full(bbox.shape[0], i + 1, dtype=np.int32)
        for i, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)
    # (xmin, ymin, xmax, ymax, score)
    bboxes = np.vstack(result)
    return (result, bboxes, labels)


def inner_nms(dets, thresh=0.2):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / areas[i]
        reverse = inter / areas[order[1:]]
        ovr = np.where(ovr > reverse, ovr, reverse)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def get_coco_test_dataset(path):
    coco = COCO(path)
    cats = coco.loadCats(coco.getCatIds())
    categories = [c['name'] for c in cats]
    # get label name

    img_cls = {}
    image_ids = coco.getImgIds()
    for id in image_ids:
        img_name = coco.loadImgs(id)[0]['file_name']
        ann_id = coco.getAnnIds(id)
        ann_info = coco.loadAnns(ann_id)
        ids = [a['category_id'] for a in ann_info]
        # get label ID
        img_cls[img_name] = ids
    return img_cls, categories


def get_predict_main_label(bboxes, labels, class_names):
    if len(bboxes) == 0:
        return class_names.index('BH') + 1
    main_label = labels[np.argmax(bboxes[:, 4])]
    return main_label


if __name__ == "__main__":
    img_cls, class_names = get_coco_test_dataset(
        '/data/T2-cell-55D07/xiaotu/test.json')
    print(class_names)
    SCORE_THRESH = 0.3
    IMG_ROOT = ''
    TARGET_RESULT = '/data/T2-cell-55D07/xiaotu/result_0903_00'
    if not os.path.exists(TARGET_RESULT):
        os.makedirs(TARGET_RESULT)
    im_lists = img_cls.keys()

    config_file = '/root/software/mmdetection/configs/csot_config/t2-55d07_xiaotu/faster_rcnn_dconv_c3-c5_r50_fpn_1x.py'
    model_path = '/data/55d07_workdir/xiaotu/faster_rcnn_dconv_c3-c5_r50_fpn_1x_0903_00/final/epoch_22.pth'

    model = init_detector(config_file, model_path)

    pred_truth = []
    pred_result = dict()

    tqdm_iter = tqdm(im_lists)
    kk = 0
    for im_name in tqdm_iter:

        print(img_cls[im_name])
        try:
            truth_main_label = img_cls[im_name][0]
        except IndexError:
            print(" ~"*50)
            print("Invalid category id: {}".format(img_cls[im_name]))
            print("Invalid category id file: {}".format(im_name))
            continue
        im_name = os.path.join(IMG_ROOT, im_name)
        img = mmcv.imread(im_name)
        time_begin = time.time()
        result, bboxes, labels = detect_image(model, img)
        keep_indices = np.where(bboxes[:, 4] > SCORE_THRESH)
        bboxes = bboxes[keep_indices]
        labels = labels[keep_indices]
        keep_indices = inner_nms(bboxes)
        bboxes = bboxes[keep_indices]
        labels = labels[keep_indices]

        time_end = time.time()
        time_cost = 1000 * (time_end - time_begin)

        predict_main_label = get_predict_main_label(
            bboxes, labels, class_names)

        inner = '_'.join((class_names[truth_main_label - 1],
                          class_names[predict_main_label - 1]))
        out_file = os.path.join(TARGET_RESULT, inner,
                                os.path.basename(im_name))

        mmcv.imshow_det_bboxes(
            img.copy(),
            bboxes,
            labels - 1,
            class_names=class_names,
            score_thr=SCORE_THRESH,
            bbox_color='green',
            text_color='green',
            thickness=1,
            font_scale=0.5,
            show=False,
            out_file=out_file)

        tqdm_iter.set_description('cost time {:.2f} ms, predict {}, truth {}'.format(
            time_cost, predict_main_label, truth_main_label))

        if truth_main_label is not None and predict_main_label is not None:
            pred_truth.append([predict_main_label, truth_main_label])

        sig_result = [[bboxes[i, :-1].tolist(), int(labels[i]), float(bboxes[i, -1])]
                      for i in range(bboxes.shape[0])]
        img_name = os.path.basename(im_name).split(".")[0]
        panel_key = img_name.split("-")[0] + "_" + img_name.split("-")[1]
        chan_key = int(img_name.split("-")[-1][-1])

        if predict_main_label != truth_main_label:
            continue

        if panel_key not in pred_result.keys():
            pred_result[panel_key] = dict()

        pred_result[panel_key].setdefault(chan_key, []).extend(sig_result)
        pred_result[panel_key]["code"] = im_name.split('/')[-2]

    pred_truth = np.array(pred_truth)
    cm = confusion_matrix(pred_truth[:, 1], pred_truth[:, 0])

    with open(TARGET_RESULT+"/result.json", "w") as fp:
        json.dump(pred_result, fp)
    print("write result to JSON file.")

    print('~' * 50)
    class_count = np.sum(cm, axis=1)
    for i in range(len(class_names)):
        print('class {:12} {:>6}/{:<6} {:8.4f}'.format(
            class_names[i],
            cm[i][i], class_count[i],
            float(cm[i][i]) / max(class_count[i], 1)))
    avg_score = float(np.trace(cm)) / np.sum(class_count)
    print('avg score : {:.4f}'.format(avg_score))
    print(cm)
