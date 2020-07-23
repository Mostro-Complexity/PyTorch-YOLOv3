from __future__ import division

from models import Darknet, YOLOV3Detector
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate
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

    config.TRAIN.NUM_WORKERS = args.num_workers

    config.TRAIN.EPOCHS = args.epochs
    config.TRAIN.BATCH_SIZE = args.batch_size
    config.TRAIN.GRADIENT_ACCUMULATIONS = args.gradient_accumulations

    config.TRAIN.PRETRAINED_WEIGHTS = args.pretrained_weights

    config.TRAIN.CHECKPOINT_INTERVAL = args.checkpoint_interval
    config.TRAIN.EVALUATION_INTERVAL = args.evaluation_interval

    config.TRAIN.PATH_TO_IMAGES_DIR = args.training_images_dir_path
    config.TRAIN.PATH_TO_ANNOTATIONS = args.training_annotations_path

    config.EVAL.BATCH_SIZE = args.batch_size
    config.EVAL.NUM_WORKERS = args.num_workers
    config.EVAL.PATH_TO_IMAGES_DIR = args.evaluating_images_dir_path
    config.EVAL.PATH_TO_ANNOTATIONS = args.evaluating_annotations_path
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--config", type=str, required=True, help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, default='', help="if specified starts from checkpoint model")
    parser.add_argument("--num_workers", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--num_classes", type=int, default=80, help="number of classes in dataset")
    parser.add_argument("--image_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=10, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--training_images_dir_path", default='data/train', type=str, help="path of folder for storing images in the data set")
    parser.add_argument("--training_annotations_path", default='data/annotations', type=str, help="path for storing label files in the data set")
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

    if torch.cuda.is_available():
        model = model.cuda()

    # If specified we start from checkpoint
    # if config.TRAIN.PRETRAINED_WEIGHTS is not None and config.TRAIN.PRETRAINED_WEIGHTS != '':
    #     if config.TRAIN.PRETRAINED_WEIGHTS.endswith(".pth"):
    #         backbone.load_state_dict(torch.load(config.TRAIN.PRETRAINED_WEIGHTS))
    #     else:
    #         backbone.load_darknet_weights(config.TRAIN.PRETRAINED_WEIGHTS)

    # Get dataloader
    dataset = DatasetBase.from_name('tiny-person')(
        config.TRAIN.PATH_TO_IMAGES_DIR, config.TRAIN.PATH_TO_ANNOTATIONS, DatasetBase.Mode.TRAIN)

    dataloader = DataLoader(
        dataset, batch_size=config.TRAIN.BATCH_SIZE,
        sampler=DatasetBase.NearestRatioRandomSampler(dataset.image_ratios, num_neighbors=config.TRAIN.BATCH_SIZE),
        num_workers=config.TRAIN.NUM_WORKERS, collate_fn=dataset.collate_fn, pin_memory=True
    )
    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(config.TRAIN.EPOCHS):
        model.train()
        start_time = time.time()
        for batch_id, (imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_id

            imgs = imgs.to(device)
            targets = targets.to(device)

            loss = model(imgs, targets)

            loss.backward()
            torch.cuda.empty_cache()

            if batches_done % config.TRAIN.GRADIENT_ACCUMULATIONS:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, config.TRAIN.EPOCHS, batch_id, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.detections))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % detection.metrics.get(metric, 0) for detection in model.detections]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.feature.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                # logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_id + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_id + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.feature.seen += imgs.size(0)

        if epoch % config.TRAIN.EVALUATION_INTERVAL == 0:
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
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

        if epoch % config.TRAIN.CHECKPOINT_INTERVAL == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
    
    # final model saving
    torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)