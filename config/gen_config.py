
import glob

import numpy as np

from sklearn.cluster import KMeans
from torchvision.datasets import CocoDetection
import argparse
import json
import torch
import yaml
import os

N_CLUSTERS = 9
IMAGE_SIZE = 416


class Annotation2Anchor(object):
    def __init__(self, desired_img_size):
        self.desired_img_size = desired_img_size

    def __call__(self, target, attribute):
        if len(target) != 0 and len(attribute) != 0:
            target = target[0]
            attribute = attribute[0]
            prior_width = target['bbox'][2]
            prior_height = target['bbox'][3]

            img_height = attribute['height']
            img_width = attribute['width']

            prior_width = prior_width / img_width * self.desired_img_size
            prior_height = prior_height / img_height * self.desired_img_size

            return np.array([prior_width, prior_height])


class CocoAnnotation(CocoDetection):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)

        target = coco.loadAnns(ann_ids)
        attribute = coco.loadImgs(img_id)

        if self.transforms is not None:
            target = self.transforms(target, attribute)

        return target


def format_anchor(config_file, output_path, prior_sizes):
    prior_sizes = prior_sizes.reshape(-1)
    with open(config_file, 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
        config['GLOBAL']['ANCHORS'] = prior_sizes.tolist()
        config = yaml.dump(config)

    with open(output_path, 'w') as f:
        f.write(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir_path", "-o", type=str, required=True, help="path to output data config file")
    parser.add_argument("--num_clusters", type=int, default=N_CLUSTERS, help="number of cpu threads to use during batch generation")
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE, help="size of each image dimension")
    parser.add_argument("--training_annotations_path", default='data/annotations', type=str, help="path for storing label files in the data set")
    parser.add_argument("--config_template", default='config/yolov3-config.yaml', type=str, help="path for config template for training")
    args = parser.parse_args()

    coco_dataset = CocoAnnotation(root=None, annFile=args.training_annotations_path, transforms=Annotation2Anchor(args.image_size))
    box_sizes = np.array([box_size for box_size in coco_dataset if box_size is not None])
    cluster_centers = KMeans(n_clusters=args.num_clusters, n_init=100).fit(box_sizes).cluster_centers_

    cluster_centers = cluster_centers[cluster_centers.sum(axis=-1).argsort()]
    cluster_centers = np.round(cluster_centers).astype(np.int64)
    format_anchor(args.config_template, os.path.join(args.output_dir_path, 'yolov3-config-tiny-person.yaml'), cluster_centers)
