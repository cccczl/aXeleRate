# -*- coding: utf-8 -*-
from ._box_match import BoxMatcher

def count_true_positives(detect_boxes, true_boxes, detect_labels=None, true_labels=None):
    """
    # Args
        detect_boxes : array, shape of (n_detected_boxes, 4)
        true_boxes : array, shape of (n_true_boxes, 4)
        detected_labels : array, shape of (n_detected_boxes,)
        true_labels :
    """
    n_true_positives = 0

    matcher = BoxMatcher(detect_boxes, true_boxes, detect_labels, true_labels)
    for i in range(len(detect_boxes)):
        matching_idx, iou = matcher.match_idx_of_box1_idx(i)
        print(f"detect_idx: {i}, true_idx: {matching_idx}, matching-score: {iou}")
        if matching_idx is not None and iou > 0.5:
            n_true_positives += 1
    return n_true_positives


def calc_score(n_true_positives, n_truth, n_pred):
    """
    # Args
        detect_boxes : list of box-arrays
        true_boxes : list of box-arrays
    """
    precision = n_true_positives / n_pred if n_pred > 0 else 0
    if n_truth > 0:
        recall = n_true_positives / n_truth
    elif n_truth == 0 and n_true_positives == 0:
        recall = 1
    else:
        recall = 0
    if precision + recall > 0:
        fscore = 2* precision * recall / (precision + recall)
        return {"fscore": fscore, "precision": precision, "recall": recall}
    else:
        return 0
    

