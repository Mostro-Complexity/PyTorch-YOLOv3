from typing import List, Tuple


def group_anchors(anchor_list: List[int], num_groups: int) -> Tuple[List[Tuple[int]]]:
    num_anchor_boxes_per_group: int = len(anchor_list)//num_groups//2
    grouped_anchors_boxes: List = []
    anchor_boxes: List = []
    for i in range(0, len(anchor_list), 2):
        anchor_boxes.append((anchor_list[i], anchor_list[i + 1]))

    for i in range(num_groups):
        grouped_anchors_boxes.append(anchor_boxes[i*num_anchor_boxes_per_group:(i + 1)*num_anchor_boxes_per_group])

    return tuple(grouped_anchors_boxes)
