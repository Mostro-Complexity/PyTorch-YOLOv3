import os
from typing import Union, Tuple, List, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from backbone.base import Base as BackboneBase
from extension.anchor import group_anchors

from utils.utils import build_targets


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class YOLOV3(nn.Module):

    def __init__(self, backbone: BackboneBase, num_classes: int, anchor_sizes: List[int],
                 model_def, image_size: Optional[int] = 416):  # TODO:删除model_def
        super(YOLOV3, self).__init__()
        # Initiate model
        self.backbone = backbone.backbone(model_def)
        self.backbone.apply(weights_init_normal)

        self.feature = backbone.features()

        anchor_group_1, anchor_group_2, anchor_group_3 = group_anchors(anchor_sizes, num_groups=3)
        self.detections = nn.ModuleList([
            YOLOV3.Detection(anchor_group_1, num_classes, image_size),
            YOLOV3.Detection(anchor_group_2, num_classes, image_size),
            YOLOV3.Detection(anchor_group_3, num_classes, image_size)
        ])

    def forward(self, image_batch: Tensor, targets_batch: Tensor = None) -> Union[Tuple[Tensor, Tensor, Tensor, Tensor],
                                                                                  Tuple[Tensor, Tensor, Tensor, Tensor]]:

        if self.training:
            cascaded_scale_features = self.feature(image_batch, targets_batch)
            total_losses = [detection(cascaded_scale_features[i], targets_batch) for i, detection in enumerate(self.detections)]
            return torch.stack(total_losses).sum()
        else:
            output = self.detections(image_batch)
            return output

    def save(self, path_to_checkpoints_dir: str, step: int, optimizer: Optimizer, scheduler: _LRScheduler) -> str:
        path_to_checkpoint = os.path.join(path_to_checkpoints_dir, f'model-{step}.pth')
        checkpoint = {
            'state_dict': self.state_dict(),
            'step': step,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        torch.save(checkpoint, path_to_checkpoint)
        return path_to_checkpoint

    def load(self, path_to_checkpoint: str, optimizer: Optimizer = None, scheduler: _LRScheduler = None) -> 'Model':
        checkpoint = torch.load(path_to_checkpoint)
        self.load_state_dict(checkpoint['state_dict'])
        step = checkpoint['step']
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return step

    class Detection(nn.Module):
        """Detection layer"""

        def __init__(self, anchors: List[Tuple[int]], num_classes: int, img_dim: Optional[int] = 416):
            super(YOLOV3.Detection, self).__init__()
            self.anchors = anchors
            self.num_anchors = len(anchors)
            self.num_classes = num_classes
            self.ignore_thres = 0.5
            self.mse_loss = nn.MSELoss()
            self.bce_loss = nn.BCELoss()
            self.obj_scale = 1
            self.noobj_scale = 100
            self.metrics = {}
            self.img_dim = img_dim
            self.grid_size = 0  # grid size

        def compute_grid_offsets(self, grid_size, cuda=True):
            self.grid_size = grid_size
            g = self.grid_size
            FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
            self.stride = self.img_dim / self.grid_size
            # Calculate offsets for each grid
            self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
            self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
            self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
            self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
            self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

        def forward(self, x, targets=None):
            # Tensors for cuda support
            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor

            num_samples = x.size(0)
            grid_size = x.size(2)

            prediction = (
                x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

            # Get outputs
            x = torch.sigmoid(prediction[..., 0])  # Center x
            y = torch.sigmoid(prediction[..., 1])  # Center y
            w = prediction[..., 2]  # Width
            h = prediction[..., 3]  # Height
            pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
            pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

            # If grid size does not match current we compute new offsets
            if grid_size != self.grid_size:
                self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

            # Add offset and scale with anchors
            pred_boxes = FloatTensor(prediction[..., :4].shape)
            pred_boxes[..., 0] = x.data + self.grid_x
            pred_boxes[..., 1] = y.data + self.grid_y
            pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
            pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

            output = torch.cat(
                (
                    pred_boxes.view(num_samples, -1, 4) * self.stride,
                    pred_conf.view(num_samples, -1, 1),
                    pred_cls.view(num_samples, -1, self.num_classes),
                ),
                -1,
            )

            if targets is None:
                return output
            else:
                iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                    pred_boxes=pred_boxes,
                    pred_cls=pred_cls,
                    target=targets,
                    anchors=self.scaled_anchors,
                    ignore_thres=self.ignore_thres,
                )

                # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
                loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
                loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
                loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
                loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
                loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
                loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
                loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
                loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
                total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

                # Metrics
                cls_acc = 100 * class_mask[obj_mask].mean()
                conf_obj = pred_conf[obj_mask].mean()
                conf_noobj = pred_conf[noobj_mask].mean()
                conf50 = (pred_conf > 0.5).float()
                iou50 = (iou_scores > 0.5).float()
                iou75 = (iou_scores > 0.75).float()
                detected_mask = conf50 * class_mask * tconf
                precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
                recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
                recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

                self.metrics = {
                    "loss": total_loss.item(),
                    "x": loss_x.item(),
                    "y": loss_y.item(),
                    "w": loss_w.item(),
                    "h": loss_h.item(),
                    "conf": loss_conf.item(),
                    "cls": loss_cls.item(),
                    "cls_acc": cls_acc.item(),
                    "recall50": recall50.item(),
                    "recall75": recall75.item(),
                    "precision": precision.item(),
                    "conf_obj": conf_obj.item(),
                    "conf_noobj": conf_noobj.item(),
                    "grid_size": grid_size,
                }

                return total_loss
