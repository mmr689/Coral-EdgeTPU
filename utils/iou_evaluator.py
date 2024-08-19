# iou_evaluator.py
from collections import Counter

def bbox_iou(box1, box2):
    """Calcula el Intersection over Union (IoU) de dos bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = map(float, box1)
    x2_min, y2_min, x2_max, y2_max = map(float, box2)
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max > inter_x_min and inter_y_max > inter_y_min:
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    else:
        inter_area = 0
    
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area else 0
    
    return iou

def evaluate_predictions(predictions, ground_truth, iou_threshold=0.5):
    """Evalúa predicciones contra el ground truth utilizando el umbral de IoU."""
    true_positives = []
    true_redundat_positives = []
    matched_truths = []
    all_matched_truths = []
    false_positives = []

    for pred in predictions:
        match_found = False
        for truth in ground_truth:
            iou = bbox_iou(pred, truth)
            if iou >= iou_threshold:
                if truth in matched_truths:
                    true_redundat_positives.append((pred, iou))
                else:
                    true_positives.append((pred, iou))
                matched_truths.append(truth)
                all_matched_truths.append([truth, pred, iou])
                match_found = True
                break
        if not match_found:
            false_positives.append((pred, iou))


    false_negatives = [truth for truth in ground_truth if truth not in matched_truths]
    
    return true_positives, true_redundat_positives, false_positives, false_negatives