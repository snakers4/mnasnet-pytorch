import os
import itertools


def calculate_iou_rate(image_id):

    bbox_list = get_bbox_list(image_id)

    bboxes = []
    for bbox_tuple in bbox_list:
        bbox_dict = bb_tuple2dict_2d(bbox_tuple)
        bboxes.append(bbox_dict)

    ious = []
    for x in itertools.product(bboxes, bboxes):
        if(x[0] == x[1]):
            pass
        else:
            ious.append(iou_2d(x[0], x[1]))

    if len(ious) == 0:
        return 0
    else:
        return sum(ious)/len(ious)


def get_bbox_list(image_id):
    global bbox_dict
    return bbox_dict[image_id]


def get_bbox_list_label(image_id):
    global bbox_dict_label
    return bbox_dict_label[image_id]


def bb_tuple2dict_2d(bbox):
    bbox_dict = {
        'x1': bbox[0][0],
        'x2': bbox[0][1],
        'y1': bbox[1][0],
        'y2': bbox[1][1]
    }
    return bbox_dict


def iou_2d(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def bbox_inside_2d(bb1, bb2):
    """
    Calculate whether bb1 is contained within bb2

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y2) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    if (bb2['x1'] < bb1['x1']
        and bb2['x2'] > bb1['x2']
        and bb2['y1'] < bb1['y1']
            and bb2['y2'] > bb1['y2']):
        return 1
    else:
        return 0

    
class bbox_2d(object):
    def __init__(self,
                 bbox_tuple
                 ):
        assert len(bbox_tuple) == 2
        assert len(bbox_tuple[0]) == 2
        assert len(bbox_tuple[1]) == 2

        self.x_min = bbox_tuple[0][0]
        self.x_max = bbox_tuple[0][1]
        self.y_min = bbox_tuple[1][0]
        self.y_max = bbox_tuple[1][1]