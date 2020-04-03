#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pip install lxml

import sys
import os
import cv2
import json
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import glob

home = os.path.expanduser("~") + "/"
fww1030_image_id = 10000000
fww1030_bounding_box_id = 10000000


def get_jpglist(path):
    file_list = glob.glob(path+"/*/*.bmp")
    print(file_list[:10])
    # file_list = [f.replace(home, "") for f in file_list]
    fp = open("jpg_list.txt", "w")
    for f in file_list:
        try:
            cv2.imread(f)
        except Exception:
            continue
        fp.write(f)
        fp.write("\n")
    fp.close()


def get_current_image_id():
    global fww1030_image_id
    fww1030_image_id += 1
    return fww1030_image_id


def get_current_annotation_id():
    global fww1030_bounding_box_id
    fww1030_bounding_box_id += 1
    return fww1030_bounding_box_id


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError(
            'The size of %s is supposed to be %d, but is %d.' % (
                name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def get_labels(xml_list):
    categories = dict()
    for line in xml_list:
        tree = ET.parse(line)
        root = tree.getroot()
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            if category not in categories:
                categories[category] = 1
            else:
                categories[category] += 1

    filted_labels = []
    for k, v in categories.items():
        print('** {} **  -> {}'.format(k, v))
        if v > 0:
            filted_labels.append(k)
    return filted_labels


def filter_image_label_path(jpg_list):
    filted_jpg_xml = []
    for jpg in jpg_list:
        if len(jpg.split('.')[-1]) == 3:
            xml = jpg[:-3] + 'xml'
        else:
            xml = jpg[:-4] + 'xml'
        if os.path.exists(jpg) and os.path.exists(xml):
            filted_jpg_xml.append((jpg, xml))
        else:
            print('No XML -> {}'.format(jpg))
    return filted_jpg_xml


def convert(jpg_xml, categories, json_file):
    json_dict = {"images": [], "type": "instances", "annotations": [],
                 "categories": []}

    for jpg_file, xml_file in jpg_xml:

        # assert(jpg_file[:-4] == xml_file[:-4])

        tree = ET.parse(xml_file)
        root = tree.getroot()

        image_id = get_current_image_id()
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)

        # 构造image
        image = {'file_name': jpg_file, 'height': height, 'width': width,
                 'id': image_id}
        json_dict['images'].append(image)

        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text

            if category not in categories:
                print('skip annotation {}'.format(category))
                continue

            category_id = categories.index(category) + 1

            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bndbox, 'xmin', 1).text) - 1
            ymin = int(get_and_check(bndbox, 'ymin', 1).text) - 1
            xmax = int(get_and_check(bndbox, 'xmax', 1).text)
            ymax = int(get_and_check(bndbox, 'ymax', 1).text)
            if (xmax <= xmin) or (ymax <= ymin):
                print('{} error'.format(filename))
                continue

            o_width = (xmax - xmin)
            o_height = (ymax - ymin)
            # image_id means the id of the image, id means the id of label
            ann = {'area': o_width*o_height, 'iscrowd': 0, 'image_id':
                   image_id, 'bbox': [xmin, ymin, o_width, o_height],
                   'category_id': category_id,
                   'id': get_current_annotation_id(), 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)

    for cid, cate in enumerate(categories):
        cat = {'supercategory': 'FWW', 'id': cid + 1, 'name': cate}
        json_dict['categories'].append(cat)

    json_fp = open(json_file, 'w')
    json.dump(json_dict, json_fp, indent=4)
    json_fp.close()
    list_fp.close()


def write_categories_2_file(categories, filename):
    if not os.path.exists(filename):
        with open(filename, 'w') as fp:
            fp.writelines([item + '\n' for item in categories])


def convert_oks_list(oks_list, categories, json_file):
    json_dict = {"images": [], "type": "instances", "annotations": [],
                 "categories": []}

    roi = cv2.imread('./template.png')

    for jpg_file in oks_list:
        current = cv2.imread(jpg_file)
        image_id = get_current_image_id()
        width = current.shape[1]
        height = current.shape[0]

        # 构造image
        image = {'file_name': jpg_file, 'height': height, 'width': width,
                 'id': image_id}
        json_dict['images'].append(image)

        for _ in range(1):
            category = 'PVH01'

            if category not in categories:
                print('skip annotation {}'.format(category))
                continue

            category_id = categories.index(category) + 1

            xmin = 100
            ymin = 100
            xmax = xmin + roi.shape[1]
            ymax = ymin + roi.shape[0]
            if (xmax <= xmin) or (ymax <= ymin):
                print('{} error'.format(filename))
                continue

            current[ymin: ymax, xmin: xmax] = roi

            o_width = (xmax - xmin)
            o_height = (ymax - ymin)
            ann = {'area': o_width*o_height, 'iscrowd': 0, 'image_id':
                   image_id, 'bbox': [xmin, ymin, o_width, o_height],
                   'category_id': category_id,
                   'id': get_current_annotation_id(), 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
        cv2.imwrite(jpg_file, current)

    for cid, cate in enumerate(categories):
        cat = {'supercategory': 'FWW', 'id': cid + 1, 'name': cate}
        json_dict['categories'].append(cat)

    json_fp = open(json_file, 'w')
    json.dump(json_dict, json_fp, indent=4)
    json_fp.close()
    list_fp.close()


if __name__ == '__main__':
    # create JPG list txt
    data_path = "/home/zy/share2-47/华星光电CSOT/t1-cell-api-43&49/train_sample/train_data_rechecked/Big"
    if not os.path.exists("./jpg_list.txt"):
        get_jpglist(data_path)
        print("*"*30 + "get jpg_list file.")
    # read JPG path from list txt
    list_fp = open('jpg_list.txt', 'r')
    jpg_list = [item.strip() for item in list_fp]
    jpg_xml = filter_image_label_path(jpg_list)
    print('\nnum samples -> ** {} **\n'.format(len(jpg_xml)))

    categories = get_labels([item[1] for item in jpg_xml])
    if os.path.exists("labels.txt"): 
        fp = open("labels.txt", "r")
        categories = fp.readlines()
        categories = [cat.replace("\n", "") for cat in categories]

    jpgs = [item[0] for item in jpg_xml]
    xmls = [item[1] for item in jpg_xml]
    print(categories)
    labels = [item.split('/')[-2] for item in jpgs]

    split = False    # 是否划分训练/测试集

    if split:
        train_x, test_x, train_y, test_y = train_test_split(
            jpgs, xmls, test_size=0.3, random_state=666, stratify=labels)
        train_samples = [(jpg, xml) for jpg, xml in zip(train_x, train_y)]
        test_samples = [(jpg, xml) for jpg, xml in zip(test_x, test_y)]

        convert(train_samples, categories, 'train.json')
        convert(test_samples, categories, 'test.json')
    else:
        convert(jpg_xml, categories, "all_sample.json")

    write_categories_2_file(categories, 'labels.txt')

    # list_fp = open('oks_list.txt', 'r')
    # jpg_list = [item.strip() for item in list_fp]
    # train_list = []
    # for i, item in enumerate(jpg_list):
    #     if i % 5 != 0:
    #         train_list.append(item)

    # print('** oks **  -> {}'.format(len(train_list)))
    # convert_oks_list(train_list, categories, 'train_oks.json')
