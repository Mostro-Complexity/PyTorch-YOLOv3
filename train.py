from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate
from extension.anchor import group_anchors
from extension.config.yolov3_config import _C as config
from terminaltables import AsciiTable

from dataset.base import Base as DatasetBase
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--config", type=str, required=True, help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--num_workers", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--num_classes", type=int, default=80, help="number of classes in dataset")
    parser.add_argument("--image_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=10, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    config.merge_from_file(args.config)

    # Initiate model
    backbone = Darknet(args.model_def).to(device)
    backbone.apply(weights_init_normal)

    # Anchor arrangement
    anchor_group_1, anchor_group_2, anchor_group_3 = group_anchors(config.GLOBAL.ANCHORS, num_groups=3)
    detector = [
        YOLOV3Detector(anchor_group_1, config.GLOBAL.NUM_CLASSES, config.GLOBAL.IMAGE_SIZE[0]),
        YOLOV3Detector(anchor_group_2, config.GLOBAL.NUM_CLASSES, config.GLOBAL.IMAGE_SIZE[0]),
        YOLOV3Detector(anchor_group_3, config.GLOBAL.NUM_CLASSES, config.GLOBAL.IMAGE_SIZE[0])
    ]
    # If specified we start from checkpoint
    if args.pretrained_weights:
        if args.pretrained_weights.endswith(".pth"):
            backbone.load_state_dict(torch.load(args.pretrained_weights))
        else:
            backbone.load_darknet_weights(args.pretrained_weights)

    # Get dataloader
    dataset = DatasetBase.from_name('tiny-person')(
        'data/tiny_set/train', 'data/tiny_set/erase_with_uncertain_dataset/annotations/corner/task/tiny_set_train_sw640_sh512_all.json', DatasetBase.Mode.TRAIN)

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=DatasetBase.NearestRatioRandomSampler(dataset.image_ratios, num_neighbors=args.batch_size),
        num_workers=args.num_workers, collate_fn=dataset.collate_fn, pin_memory=True
    )
    optimizer = torch.optim.Adam(backbone.parameters())

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

    for epoch in range(args.epochs):
        backbone.train()
        start_time = time.time()
        for batch_id, (imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_id

            imgs = imgs.to(device)
            targets = targets.to(device)

            multiscale_features = backbone(imgs, targets)

            loss = detector[0](multiscale_features[0], targets, 416)
            loss += detector[1](multiscale_features[1], targets, 416)
            loss += detector[2](multiscale_features[2], targets, 416)

            loss.backward()
            torch.cuda.empty_cache()

            if batches_done % args.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, args.epochs, batch_id, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(backbone.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in detector]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(backbone.yolo_layers):
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

            backbone.seen += imgs.size(0)

        # if epoch % opt.evaluation_interval == 0:
        #     print("\n---- Evaluating Model ----")
        #     # Evaluate the model on the validation set
        #     precision, recall, AP, f1, ap_class = evaluate(
        #         model,
        #         path=valid_path,
        #         iou_thres=0.5,
        #         conf_thres=0.5,
        #         nms_thres=0.5,
        #         img_size=opt.img_size,
        #         batch_size=8,
        #     )
        #     evaluation_metrics = [
        #         ("val_precision", precision.mean()),
        #         ("val_recall", recall.mean()),
        #         ("val_mAP", AP.mean()),
        #         ("val_f1", f1.mean()),
        #     ]
        #     logger.list_of_scalars_summary(evaluation_metrics, epoch)

        #     # Print class APs and mAP
        #     ap_table = [["Index", "Class name", "AP"]]
        #     for i, c in enumerate(ap_class):
        #         ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
        #     print(AsciiTable(ap_table).table)
        #     print(f"---- mAP {AP.mean()}")

        if epoch % args.checkpoint_interval == 0:
            torch.save(backbone.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
