# iou_evaluator.py

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
    valid_predictions = []
    matched_truths = []
    false_positives = []

    for pred in predictions:
        match_found = False
        for truth in ground_truth:
            iou = bbox_iou(pred, truth)
            if iou >= iou_threshold:
                valid_predictions.append((pred, iou))
                matched_truths.append(truth)
                match_found = True
                break
        if not match_found:
            false_positives.append((pred, iou))
    
    false_negatives = [truth for truth in ground_truth if truth not in matched_truths]
    
    return valid_predictions, false_positives, false_negatives