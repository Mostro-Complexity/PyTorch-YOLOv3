from __future__ import division

from models import Darknet, YOLOV3Detector
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from extension.anchor import group_anchors
from extension.config.yolov3_config import _C as config
from terminaltables import AsciiTable

from dataset.base import Base as DatasetBase
from backbone.base import Base as BackboneBase
from model import YOLOV3
import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def setup_config(config, args):

    config.GLOBAL.IMAGE_SIZE = (args.image_size, args.image_size)
    config.GLOBAL.NUM_CLASSES = args.num_classes

    config.EVAL.BATCH_SIZE = args.batch_size
    config.EVAL.NUM_WORKERS = args.num_workers
    config.EVAL.PATH_TO_IMAGES_DIR = args.evaluating_images_dir_path
    config.EVAL.PATH_TO_ANNOTATIONS = args.evaluating_annotations_path
    return config


def evaluate(model, path_to_images_dir: str, path_to_annotation: str, iou_thres, conf_thres, nms_thres, img_size, batch_size, num_workers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Get dataloader
    dataset = DatasetBase.from_name('tiny-person')(
        'data/tiny_set/train', 'data/tiny_set/erase_with_uncertain_dataset/annotations/corner/task/tiny_set_train_sw640_sh512_all.json', DatasetBase.Mode.TRAIN)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for imgs, targets in tqdm.tqdm(dataloader, desc="Detecting objects"):
        imgs = imgs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [torch.cat(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class, dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--config", type=str, required=True, help="path to data config file")
    parser.add_argument("--checkpoint", type=str, default='checkpoints/yolov3_ckpt_90.pth', help="path for checkpoint model")
    parser.add_argument("--num_workers", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--num_classes", type=int, default=80, help="number of classes in dataset")
    parser.add_argument("--image_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--evaluating_images_dir_path", default='data/val', type=str, help="path of folder for storing images in the data set")
    parser.add_argument("--evaluating_annotations_path", default='data/annotations', type=str, help="path for storing label files in the data set")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    config.merge_from_file(args.config)
    config = setup_config(config, args)
    config.freeze()

    # Initiate model
    backbone = BackboneBase.from_name('darknet53')(pretrained=False, models_dir='')
    model = YOLOV3(backbone, config.GLOBAL.NUM_CLASSES, config.GLOBAL.ANCHORS, args.model_def, image_size=config.GLOBAL.IMAGE_SIZE[0])
    model.load_state_dict(torch.load(args.checkpoint))

    if torch.cuda.is_available():
        model = model.cuda()

    # If specified we start from checkpoint
    # if config.TRAIN.PRETRAINED_WEIGHTS is not None and config.TRAIN.PRETRAINED_WEIGHTS != '':
    #     if config.TRAIN.PRETRAINED_WEIGHTS.endswith(".pth"):
    #         backbone.load_state_dict(torch.load(config.TRAIN.PRETRAINED_WEIGHTS))
    #     else:
    #         backbone.load_darknet_weights(config.TRAIN.PRETRAINED_WEIGHTS)

    # Evaluate the model on the validation set
    precision, recall, AP, f1, ap_class, dataset = evaluate(
        model,
        path_to_images_dir=config.EVAL.PATH_TO_IMAGES_DIR,
        path_to_annotation=config.EVAL.PATH_TO_ANNOTATIONS,
        iou_thres=0.5,
        conf_thres=0.5,
        nms_thres=0.5,
        img_size=config.GLOBAL.IMAGE_SIZE,
        batch_size=config.EVAL.BATCH_SIZE,
        num_workers=config.EVAL.NUM_WORKERS
    )
    evaluation_metrics = [
        ("val_precision", precision.mean()),
        ("val_recall", recall.mean()),
        ("val_mAP", AP.mean()),
        ("val_f1", f1.mean()),
    ]
    # logger.list_of_scalars_summary(evaluation_metrics, epoch)

    # Print class APs and mAP
    ap_table = [["Index", "Class name", "AP"]]
    for i, c in enumerate(ap_class):
        ap_table += [[c, dataset.class_names[c], "%.5f" % AP[i]]]
    print(AsciiTable(ap_table).table)
    print(f"---- mAP {AP.mean()}")
