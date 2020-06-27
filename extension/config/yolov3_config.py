from yacs.config import CfgNode as CN


_C = CN()
_C.GLOBAL = CN()
_C.GLOBAL.ANCHORS = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
_C.GLOBAL.IMAGE_SIZE = (416, 416)
_C.GLOBAL.NUM_CLASSES = 80

_C.TRAIN = CN()
_C.TRAIN.NUM_GPUS = 4
_C.TRAIN.NUM_WORKERS = 4

_C.TRAIN.EPOCHS = 100
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.GRADIENT_ACCUMULATIONS = 2

_C.TRAIN.PRETRAINED_WEIGHTS = ""

_C.TRAIN.CHECKPOINT_INTERVAL = 10
_C.TRAIN.EVALUATION_INTERVAL = 10


def get_cfg_defaults():
    return _C.clone()


def save_to_file(path):
    with open(path, 'w') as f:
        f.write(_C.dump())


if __name__ == "__main__":
    save_to_file('yolov3-config-tiny-persion.yaml')
