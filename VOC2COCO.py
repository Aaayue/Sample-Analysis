import os
import sys
import xml.etree.ElementTree as ET
import glob
import json
import logging
import datetime
from collections import Counter


def out_log(logFilename):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(filename)s : %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %A %H:%M:%S",
        filename=logFilename,
        filemode="a",
    )
    my_handler = logging.StreamHandler()
    my_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s  %(filename)s : %(levelname)s  %(message)s")
    my_handler.setFormatter(formatter)

    logging.getLogger().addHandler(my_handler)


def add_code_name(path):
    for root, dirs, files in os.walk(path, topdown=False):
        code = os.path.split(root)[-1]
        for name in files:
            new_name = code + '_' + name
            os.rename(os.path.join(root, name), os.path.join(root, new_name))


def sample_distribution(json_file):
    with open(json_file) as f:
        samples = json.load(f)
    categories = samples["categories"]
    categories = [dic["name"] for dic in categories]
    annotations = samples["annotations"]
    images = samples["images"]
    area = [x["area"] for x in annotations]
    area = Counter(area)
    box_ratio = [round(x["bbox"][-1]/x["bbox"][-2]) if (x["bbox"][-1]/x["bbox"][-2])
                 > 1 else round(x["bbox"][-1]/x["bbox"][-2], 2) for x in annotations]
    box_ratio = dict(Counter(box_ratio))
    category = [categories[x["category_id"]-1] for x in annotations]
    num_sample = len(category)
    category = dict(Counter(category))
    code_ratio = [x["file_name"].split("/")[-2] for x in images]
    code_ratio = dict(Counter(code_ratio))
    for k, v in category.items():
        r = v/num_sample
        category[k] = [v, "%.2f%%" % (r*100)]

    box_ratio_list = []
    for k, v in box_ratio.items():
        r = v/num_sample
        box_ratio[k] = [v, "%.2f%%" % (r*100)]
        box_ratio_list.append([k, v, "%.2f%%" % (r*100)])
    box_ratio_list = sorted(box_ratio_list, key=lambda s: s[1], reverse=True)

    for k, v in code_ratio.items():
        r = v/num_sample
        code_ratio[k] = [v, "%.2f%%" % (r*100)]

    logging.info(
        ("\n{:<20s} \n>>>{:>10s} \n"*4).format(
            "LABELS: ", str(categories),
            "CATEGORY DISTRIBUTION: ", str(category),
            "BBOX RATIO(h/w): ", str(box_ratio_list),
            "CODE RATIO: ", str(code_ratio),)
    )


if __name__ == "__main__":
    out_log("./datu_sample.log")   # 代码运行目录
    logging.info("="*60 + "\n")
    logging.info(datetime.datetime.now().strftime("%Y-%m-%d %A %H:%M:%S"))
    sample_distribution("/home/zy/Work/Sample_Analysis/all_sample.json")
