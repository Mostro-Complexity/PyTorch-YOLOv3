import json
import os
import pickle
import random
from typing import List, Tuple, Dict

import torch
import torch.utils.data.dataset
from PIL import Image, ImageOps
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import Tensor
from torch.nn import functional as F
from torchvision.datasets import CocoDetection
from torchvision.transforms import transforms
from tqdm import tqdm

from bbox import BBox
from dataset.base import Base
from io import StringIO
import sys


class TinyPerson(Base):

    class Annotation(object):
        class Object(object):
            def __init__(self, bbox: BBox, label: int):
                super().__init__()
                self.bbox = bbox
                self.label = label

            def __repr__(self) -> str:
                return 'Object[label={:d}, bbox={!s}]'.format(
                    self.label, self.bbox)

        def __init__(self, filename: str, objects: List[Object], corner: BBox):
            super().__init__()
            self.filename = filename
            self.objects = objects
            self.corner = corner

    def __init__(self, path_to_jpeg_images_dir: str, path_to_annotations: str, mode: Base.Mode):
        path_to_data_dir = os.path.dirname(path_to_jpeg_images_dir)
        super().__init__(path_to_data_dir, mode)

        path_to_caches_dir = os.path.join('caches', 'tiny-person', f'{self._mode.value}')
        path_to_image_ids_pickle = os.path.join(path_to_caches_dir, 'image-ids.pkl')
        path_to_image_id_dict_pickle = os.path.join(path_to_caches_dir, 'image-id-dict.pkl')
        path_to_image_ratios_pickle = os.path.join(path_to_caches_dir, 'image-ratios.pkl')

        if self._mode == TinyPerson.Mode.TRAIN:
            pass
        elif self._mode == TinyPerson.Mode.EVAL:
            pass
        else:
            raise ValueError('invalid mode')

        coco_dataset = CocoDetection(root=path_to_jpeg_images_dir, annFile=path_to_annotations)

        if os.path.exists(path_to_image_ids_pickle) and os.path.exists(path_to_image_id_dict_pickle):
            print('loading cache files...')

            with open(path_to_image_ids_pickle, 'rb') as f:
                self._image_ids = pickle.load(f)

            with open(path_to_image_id_dict_pickle, 'rb') as f:
                self._image_id_to_annotation_dict = pickle.load(f)

            with open(path_to_image_ratios_pickle, 'rb') as f:
                self._image_ratios = pickle.load(f)
        else:
            print('generating cache files...')

            os.makedirs(path_to_caches_dir, exist_ok=True)

            self._image_ids: List[int] = []
            self._image_id_to_annotation_dict: Dict[str, TinyPerson.Annotation] = {}
            self._image_ratios = []

            self._class_names = [coco_dataset.coco.cats[i + 1]['name'] for i in range(len(coco_dataset.coco.cats))]

            for idx, (image, annotation) in enumerate(tqdm(coco_dataset)):
                if len(annotation) > 0:
                    image_id = annotation[0]['image_id']  # all image_id in annotation are the same
                    self._image_ids.append(image_id)
                    img_info = coco_dataset.coco.loadImgs(image_id)[0]
                    self._image_id_to_annotation_dict[image_id] = TinyPerson.Annotation(
                        filename=os.path.join(path_to_jpeg_images_dir, img_info['file_name']),
                        objects=[TinyPerson.Annotation.Object(
                            bbox=BBox(  # `ann['bbox']` is in the format [left, top, width, height]
                                left=ann['bbox'][0],
                                top=ann['bbox'][1],
                                right=ann['bbox'][0] + ann['bbox'][2],
                                bottom=ann['bbox'][1] + ann['bbox'][3]
                            ),
                            label=ann['category_id'])
                            for ann in annotation],
                        corner=BBox(
                            left=img_info['corner'][0],
                            top=img_info['corner'][1],
                            right=img_info['corner'][2],
                            bottom=img_info['corner'][3]
                        )
                    )

                    ratio = float(image.width / image.height)
                    self._image_ratios.append(ratio)

            with open(path_to_image_ids_pickle, 'wb') as f:
                pickle.dump(self._image_ids, f)

            with open(path_to_image_id_dict_pickle, 'wb') as f:
                pickle.dump(self._image_id_to_annotation_dict, f)

            with open(path_to_image_ratios_pickle, 'wb') as f:
                pickle.dump(self.image_ratios, f)

    def __len__(self) -> int:
        return len(self._image_id_to_annotation_dict)

    def __getitem__(self, index: int) -> Tuple[str, Tensor, Tensor, Tensor, Tensor]:
        image_id = self._image_ids[index]
        annotation = self._image_id_to_annotation_dict[image_id]

        bboxes = [obj.bbox.tolist() for obj in annotation.objects]
        labels = [obj.label for obj in annotation.objects]

        bboxes = torch.tensor(bboxes, dtype=torch.float)  # `bboxes` is in the format [left, top, right, bottom]
        labels = torch.tensor(labels, dtype=torch.long)

        image = Image.open(annotation.filename).convert('RGB')  # for some grayscale images
        image = image.crop(annotation.corner.tolist())
        image_size = image.size
        # random flip on only training mode
        if self._mode == TinyPerson.Mode.TRAIN and random.random() > 0.5:
            image = ImageOps.mirror(image)
            bboxes[:, [0, 2]] = image.width - bboxes[:, [2, 0]]  # index 0 and 2 represent `left` and `right` respectively

        # image, scale = TinyPerson.preprocess_with_size(image, (416, 416))
        image = TinyPerson.preprocess(image, pad_value=0, size=(416, 416))
        bboxes = TinyPerson.adjust_bbox(bboxes, image_size)

        return image_id, image, bboxes, labels

    def evaluate(self, path_to_results_dir: str, image_ids: List[str], bboxes: List[List[float]], classes: List[int], probs: List[float]) -> Tuple[float, str]:
        self._write_results(path_to_results_dir, image_ids, bboxes, classes, probs)

        annType = 'bbox'
        path_to_coco_dir = os.path.join(self._path_to_data_dir, 'COCO')
        path_to_annotations_dir = os.path.join(path_to_coco_dir, 'annotations')
        path_to_annotation = os.path.join(path_to_annotations_dir, 'instances_val2017.json')

        cocoGt = COCO(path_to_annotation)
        cocoDt = cocoGt.loadRes(os.path.join(path_to_results_dir, 'results.json'))

        cocoEval = COCOeval(cocoGt, cocoDt, annType)
        cocoEval.evaluate()
        cocoEval.accumulate()

        original_stdout = sys.stdout
        string_stdout = StringIO()
        sys.stdout = string_stdout
        cocoEval.summarize()
        sys.stdout = original_stdout

        mean_ap = cocoEval.stats[0].item()  # stats[0] records AP@[0.5:0.95]
        detail = string_stdout.getvalue()

        return mean_ap, detail

    def _write_results(self, path_to_results_dir: str, image_ids: List[str], bboxes: List[List[float]], classes: List[int], probs: List[float]):
        results = []
        for image_id, bbox, cls, prob in zip(image_ids, bboxes, classes, probs):
            results.append(
                {
                    'image_id': int(image_id),  # COCO evaluation requires `image_id` to be type `int`
                    'category_id': cls,
                    'bbox': [   # format [left, top, width, height] is expected
                        bbox[0],
                        bbox[1],
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1]
                    ],
                    'score': prob
                }
            )

        with open(os.path.join(path_to_results_dir, 'results.json'), 'w') as f:
            json.dump(results, f)

    @property
    def image_ratios(self) -> List[float]:
        return self._image_ratios

    @property
    def num_classes(self) -> int:
        return len(self._class_names)

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    @staticmethod
    def preprocess(image: Image, pad_value: int, size: Tuple[int, int]) -> Tuple[Tensor, Tuple[int, int, int, int]]:
        dim_diff = abs(image.height - image.width)
        # (upper / left) padding and (lower / right) padding
        margin_1, margin_2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        margin = (0, 0, margin_1, margin_2) if image.height <= image.width else (margin_1, margin_2, 0, 0)
        # Transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image)
        # Add padding
        image = F.pad(image, margin, "constant", value=pad_value)
        # Resize
        image = F.interpolate(image.unsqueeze(dim=0), size, mode='bilinear').squeeze()

        return image

    @staticmethod
    def adjust_bbox(bboxes: Tensor, size: Tuple[int, int]) -> Tuple[Tensor, Tuple[int, int, int, int]]:
        original_width, original_height = size
        padded_width, padded_height = max(size), max(size)
        dim_diff = abs(original_height - original_width)
        # (upper / left) padding and (lower / right) padding
        margin_1, margin_2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        margin = (0, 0, margin_1, margin_2) if original_height <= original_width else (margin_1, margin_2, 0, 0)
        # Extract coordinates for unpadded + unscaled image.
        # Output: (x1, y1, x2, y2) input range: normal size
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
        # Adjust for added padding. Output: (x1, y1, x2, y2) input range: normal size
        x1 += margin[0]
        y1 += margin[2]
        x2 += margin[1]
        y2 += margin[3]
        # Returns (x_center, y_center, w, h)  (x_center, y_center, x2, y2) range: 0,1
        bboxes_for_yolo = torch.empty_like(bboxes)
        bboxes_for_yolo[:, 0] = ((x1 + x2) / 2) / padded_width
        bboxes_for_yolo[:, 1] = ((y1 + y2) / 2) / padded_height
        bboxes_for_yolo[:, 2] = (x2 - x1) / padded_width
        bboxes_for_yolo[:, 3] = (y2 - y1) / padded_height

        return bboxes_for_yolo

    @staticmethod
    def collate_fn(batch: List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[Tensor, Tensor]:
        targets_batch: List[torch.Tensor] = [None] * len(batch)
        image_batch: List[torch.Tensor] = [None] * len(batch)
        for i, (_, image, bboxes, labels) in enumerate(batch):
            labels = labels.unsqueeze(dim=-1)
            bid = i * torch.ones_like(labels)  # Add sample index to targets
            target = torch.cat((bid.to(torch.float32), labels.to(torch.float32), bboxes), dim=-1)
            targets_batch[i] = target
            image_batch[i] = image

        padded_image_batch = image_batch

        image_batch = torch.stack(padded_image_batch, dim=0)
        targets_batch = torch.cat(targets_batch, dim=0)
        return image_batch, targets_batch

    @staticmethod
    def padding_collate_fn(batch: List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[Tensor, Tensor]:
        targets_batch: List[torch.Tensor] = [None] * len(batch)
        image_batch: List[torch.Tensor] = [None] * len(batch)
        for i, (_, image, bboxes, labels) in enumerate(batch):
            labels = labels.unsqueeze(dim=-1)
            bid = i * torch.ones_like(labels)  # Add sample index to targets
            target = torch.cat((bid.to(torch.float32), labels.to(torch.float32), bboxes), dim=-1)
            targets_batch[i] = target
            image_batch[i] = image

        max_targets_length = max([len(it) for it in targets_batch])
        padded_image_batch = image_batch
        padded_targets_batch = []

        for target in targets_batch:
            padded_targets = torch.cat([target, torch.zeros(max_targets_length - len(target), 6).to(target)])
            padded_targets_batch.append(padded_targets)

        padded_image_batch = torch.stack(padded_image_batch, dim=0)
        padded_targets_batch = torch.stack(padded_targets_batch, dim=0)
        return padded_image_batch, padded_targets_batch
